"""Environment-Driven Closed-Loop Simulation Runner.

Executes complete coarse-and-fine PAT missions within arbitrary user-defined
or preset virtual environments with sub-micro-radian tracking accuracy.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fsoc_pat.control.dual_stage import DualStageController, DualStagePointingState, FSMConfig
from fsoc_pat.control.pid import PIDConfig, PIDController
from fsoc_pat.core.state import CameraState
from fsoc_pat.core.types import AngularError, AngularState, Orientation, PixelCoord, Position3D, Velocity3D
from fsoc_pat.disturbances.atmospheric import AtmosphericConfig, AtmosphericTurbulence
from fsoc_pat.disturbances.platform_jitter import HarmonicTone, JitterConfig, PlatformJitter
from fsoc_pat.environments.models import AtmosphereModel, EnvironmentDefinition
from fsoc_pat.estimation.adaptive_ekf import AdaptiveEKF, AdaptiveEKFConfig
from fsoc_pat.estimation.imm import IMMFilter
from fsoc_pat.geometry.angular_error import compute_angular_error_from_pixel
from fsoc_pat.geometry.camera_model import CameraIntrinsics
from fsoc_pat.geometry.projection import project_point_to_pixel
from fsoc_pat.link.budget import FSOCLinkBudget
from fsoc_pat.link.link_types import LaserConfig, OpticalReceiverConfig
from fsoc_pat.pat.pat_state_machine import PATStateMachine
from fsoc_pat.pat.pat_types import PATConfig, PATState
from fsoc_pat.perception.adaptive_detector import AdaptiveDetector, AdaptiveDetectorConfig
from fsoc_pat.perception.renderer import RenderConfig, SyntheticRenderer
from fsoc_pat.simulation.beacon import BeaconProperties
from fsoc_pat.simulation.dynamics import GimbalDynamics, VirtualGimbal
from fsoc_pat.simulation.trajectories import CircularTrajectory
from fsoc_pat.telemetry.engine import (
    CSVTelemetrySink,
    JSONLTelemetrySink,
    MemoryBufferSink,
    TelemetryEngine,
    UDPSink,
)


@dataclass
class EnvironmentRunResults:
    """Summary of a completed environment simulation run.

    Attributes:
        environment_name: Name of the executed environment.
        duration_s: Total simulated time.
        steps_executed: Total simulation steps.
        wall_time_s: Elapsed real execution wall-clock time.
        coarse_rms_urad: Root-mean-square coarse gimbal pointing error (urad).
        fine_rms_urad: Root-mean-square fine optical pointing error after FSM (urad).
        mean_snr_db: Mean communication link SNR (dB).
        fine_lock_availability_pct: Percentage of time in diffraction-limited fine lock.
        summary_stats: Detailed dictionary of recorded telemetry statistics.
    """

    environment_name: str
    duration_s: float
    steps_executed: int
    wall_time_s: float
    coarse_rms_urad: float
    fine_rms_urad: float
    mean_snr_db: float
    fine_lock_availability_pct: float
    summary_stats: dict[str, Any]


class EnvironmentSimulationRunner:
    """End-to-end simulator executing user-defined virtual environments."""

    def __init__(
        self,
        environment: EnvironmentDefinition,
        duration_s: float = 6.0,
        dt: float = 0.02,
        seed: int = 42,
    ) -> None:
        self.environment = environment
        self.duration_s = duration_s
        self.dt = dt
        self.seed = seed
        self.telemetry = TelemetryEngine(environment.telemetry_schema)

    def run(
        self,
        csv_output_path: str | Path | None = None,
        jsonl_output_path: str | Path | None = None,
        udp_port: int | None = None,
    ) -> EnvironmentRunResults:
        """Execute closed-loop tracking simulation in this virtual environment.

        Args:
            csv_output_path: Optional path to stream telemetry CSV.
            jsonl_output_path: Optional path to stream JSON-lines.
            udp_port: Optional UDP port for real-time telemetry streaming.

        Returns:
            EnvironmentRunResults with statistical metrics and telemetry summary.
        """
        t_wall_start = time.perf_counter()
        n_steps = int(self.duration_s / self.dt)

        # 1. Register Sinks
        if csv_output_path:
            self.telemetry.add_sink(CSVTelemetrySink(csv_output_path))
        if jsonl_output_path:
            self.telemetry.add_sink(JSONLTelemetrySink(jsonl_output_path))
        if udp_port:
            self.telemetry.add_sink(UDPSink(port=udp_port))

        self.telemetry.open()

        # 2. Configure Subsystems based on EnvironmentDefinition
        env = self.environment
        intrinsics = CameraIntrinsics.from_fov(640, 480, math.radians(45.0))

        # Synthetic sensor renderer with environment noise & background clutter
        render_cfg = RenderConfig(
            noise_std=env.disturbances.sensor_read_noise_std,
            shot_noise=True,
            num_stars=env.disturbances.background_clutter_count,
            enable_bloom=True,
            seed=self.seed,
        )
        renderer = SyntheticRenderer(intrinsics, render_cfg)
        detector = AdaptiveDetector()

        # Estimator: Adaptive EKF with turbulence coupling
        ekf = AdaptiveEKF(AdaptiveEKFConfig(
            base_r_std=env.disturbances.sensor_read_noise_std * 0.35,
            q_accel_std=400.0,
            gate_threshold=25.0,
            scintillation_coupling=2.0,
        ))

        # Dual-Stage Pointing Controller (Coarse Gimbal + FSM)
        fsm_cfg = FSMConfig(
            max_deflection_urad=env.payload.fsm_max_deflection_urad,
            bandwidth_hz=env.payload.fsm_bandwidth_hz,
            resolution_urad=0.05,
            offload_gain=2.5,
            offload_threshold_urad=300.0,
        )
        gimbal_ctrl = PIDController(PIDConfig(kp=8.0, ki=2.0, kd=0.1, output_limit=2.0))
        dual_stage = DualStageController(
            gimbal_controller=gimbal_ctrl,
            fsm_config=fsm_cfg,
            fsm_p_gain=1.0,
            fsm_i_gain=0.0,
        )

        # Gimbal Dynamics
        gimbal = VirtualGimbal(GimbalDynamics(
            pan_rate_limit=math.radians(60.0),
            tilt_rate_limit=math.radians(45.0),
            actuator_time_constant=0.015,
        ))

        # Disturbances
        tones_pan = [
            HarmonicTone(frequency_hz=f, amplitude_rad=a * 1e-6)
            for f, a in zip(env.platform.jitter_frequencies_hz, env.platform.jitter_amplitudes_urad)
        ]
        tones_tilt = [
            HarmonicTone(frequency_hz=f * 1.1, amplitude_rad=a * 0.8 * 1e-6)
            for f, a in zip(env.platform.jitter_frequencies_hz, env.platform.jitter_amplitudes_urad)
        ]

        jitter = PlatformJitter(JitterConfig(
            tones_pan=tones_pan,
            tones_tilt=tones_tilt,
            random_std_rad=env.platform.wind_gust_buffet_std_urad * 1e-6,
            seed=self.seed,
        ))

        atmos = AtmosphericTurbulence(AtmosphericConfig(
            cn2=env.atmosphere.cn2,
            link_distance_m=1000.0,
            aperture_diameter_m=env.payload.rx_aperture_diameter_m,
            scintillation_index=env.atmosphere.scintillation_index,
            seed=self.seed + 1,
        ))

        # Optical Link Budget
        laser_cfg = LaserConfig(
            power_w=env.payload.laser_power_mw * 1e-3,
            wavelength_m=env.payload.laser_wavelength_nm * 1e-9,
        )
        rx_cfg = OpticalReceiverConfig(
            aperture_diameter_m=env.payload.rx_aperture_diameter_m,
        )
        link_budget = FSOCLinkBudget(laser_cfg, rx_cfg)

        # PAT State Machine
        pat_sm = PATStateMachine(PATConfig())

        # Target trajectory (3D circular pass representing mission dynamics)
        trajectory = CircularTrajectory(
            center=Position3D(500.0, 0.0, 0.0),
            radius=25.0,
            angular_rate=0.4,
        )
        beacon_prop = BeaconProperties(
            intensity=240.0,
            size_px=6.0,
        )

        # Pre-point gimbal to initial target LOS for rapid coarse acquisition
        p0 = trajectory.position(0.0)
        init_pan = math.atan2(p0.y, p0.x)
        init_tilt = math.atan2(-p0.z, math.hypot(p0.x, p0.y))
        gimbal.set_position(init_pan, init_tilt)

        sim_time = 0.0
        diffraction_limit_rad = env.payload.diffraction_limit_urad * 1e-6

        # --- Closed-Loop Simulation Stepping ---
        for step in range(n_steps):
            t = sim_time

            # 1. Advance Ground Truth Target
            target_pos = trajectory.position(t)
            target_vel = trajectory.velocity(t)
            target_range = math.sqrt(target_pos.x**2 + target_pos.y**2 + target_pos.z**2)
            target_az = math.atan2(target_pos.y, target_pos.x)
            target_el = math.atan2(-target_pos.z, math.hypot(target_pos.x, target_pos.y))

            # 2. Camera Platform Motion + Atmospheric Jitter
            jit_pan, jit_tilt = jitter.compute(t, self.dt)
            sway_rad = math.radians(env.platform.sway_amplitude_deg) * math.sin(2 * math.pi * env.platform.sway_frequency_hz * t)

            cam_state = CameraState(
                position=Position3D(0.0, 0.0, 0.0),
                orientation=Orientation(yaw=0.0, pitch=sway_rad, roll=0.0),
                angular_state=AngularState(
                    pan=gimbal.state.pan + jit_pan,
                    tilt=gimbal.state.tilt + jit_tilt,
                    pan_rate=gimbal.state.pan_rate,
                    tilt_rate=gimbal.state.tilt_rate,
                ),
                timestamp=t,
            )

            # 3. Render Synthetic Sensor Image
            img = renderer.render(
                cam_state,
                [(target_pos.as_array(), beacon_prop)],
            )

            # 4. Environment-Adaptive Perception Centroiding
            det = detector.detect(img)
            meas_conf = det.confidence if det.detected else 0.0

            # 5. Adaptive EKF Estimation
            ekf.predict(self.dt)
            if det.detected:
                est_u, est_v = ekf.update(det, scintillation_index=env.atmosphere.scintillation_index)
            else:
                est_u, est_v = ekf.estimated_pixel

            # Compute angular pointing error from estimate
            est_angular_err = compute_angular_error_from_pixel(
                PixelCoord(u=est_u, v=est_v),
                intrinsics,
            )

            # Control error coordinates: In standard CV, image V is down (+Z).
            # Centering a positive V error requires negative gimbal tilt rate / negative FSM tilt.
            ctrl_angular_err = AngularError(
                horizontal=est_angular_err.horizontal,
                vertical=-est_angular_err.vertical,
                total=est_angular_err.total,
            )

            # 6. Dual-Stage Coarse + FSM Control
            gimbal_cmd, dual_state = dual_stage.compute(ctrl_angular_err, self.dt)

            # 7. Supervisory PAT State Machine
            pat_pan_cmd, pat_tilt_cmd = pat_sm.step(
                dt=self.dt,
                detected=det.detected,
                angular_error=est_angular_err,
                gimbal_pan=gimbal.state.pan,
                gimbal_tilt=gimbal.state.tilt,
                pid_cmd=gimbal_cmd,
                timestamp=t,
            )
            pat_state = pat_sm.state

            # Drive Gimbal Dynamics
            gimbal.step(pat_pan_cmd, pat_tilt_cmd, self.dt)

            # 8. FSOC Optical Link Evaluation
            fine_err_rad = dual_state.fine_error_rad
            in_fine_lock = fine_err_rad <= diffraction_limit_rad

            link_res = link_budget.evaluate(
                distance_m=target_range,
                pointing_error_rad=fine_err_rad,
            )

            # 9. Log Custom Telemetry Packet
            rec: dict[str, Any] = {
                "time_s": t,
                "step": step,
                "target_range_m": target_range,
                "target_azimuth_rad": target_az,
                "target_elevation_rad": target_el,
                "gimbal_pan_rad": gimbal.state.pan,
                "gimbal_tilt_rad": gimbal.state.tilt,
                "gimbal_rate_pan_rad_s": gimbal.state.pan_rate,
                "gimbal_rate_tilt_rad_s": gimbal.state.tilt_rate,
                "fsm_pan_urad": dual_state.fsm_pan_urad,
                "fsm_tilt_urad": dual_state.fsm_tilt_urad,
                "fsm_offload_active": dual_state.offload_active,
                "coarse_error_urad": dual_state.coarse_error_urad,
                "fine_error_urad": dual_state.fine_error_urad,
                "in_fine_lock": in_fine_lock,
                "beacon_detected": det.detected,
                "detection_confidence": meas_conf,
                "est_pixel_u": est_u,
                "est_pixel_v": est_v,
                "received_power_dbm": link_res.rx_power_dbm,
                "link_snr_db": link_res.snr_db,
                "bit_error_rate": link_res.ber,
                "pat_state": pat_state.name,
            }

            self.telemetry.record(rec)
            sim_time += self.dt

        self.telemetry.close()
        t_wall_elapsed = time.perf_counter() - t_wall_start

        # Compute summary metrics
        stats = self.telemetry.get_summary()
        coarse_rms = stats.get("coarse_error_rms_urad", 0.0)
        fine_rms = stats.get("fine_error_rms_urad", 0.0)
        mean_snr = stats.get("mean_snr_db", 0.0)
        lock_avail = stats.get("fine_lock_availability_pct", 0.0)

        return EnvironmentRunResults(
            environment_name=env.name,
            duration_s=self.duration_s,
            steps_executed=n_steps,
            wall_time_s=t_wall_elapsed,
            coarse_rms_urad=coarse_rms,
            fine_rms_urad=fine_rms,
            mean_snr_db=mean_snr,
            fine_lock_availability_pct=lock_avail,
            summary_stats=stats,
        )
