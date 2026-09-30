"""Video ingestion and PTZ camera bypass engine for PAT Benchmark Performance-2.

This module allows OptiLign to:
1. Ingest arbitrary external MP4 video files (@30 FPS or any frame rate).
2. Bypass the virtual 3D camera / PTZ gimbal and feed video frames directly into
   the computer vision and AI beacon detection pipeline.
3. Calculate precision centroiding metrics:
   - Instantaneous pixel tracking error (px)
   - Cumulative Root-Mean-Square Error (RMSE in px)
   - Acquisition Time (<= 2 sec)
   - Re-acquisition Time (<= 1 sec)
   - Lock Retention Rate (Target Loss < 5%)
   - Real-time Processing Speed (>= 20 FPS)
4. Generate synthetic benchmark MP4 videos with ground truth CSV
   for self-test and verification.
"""

from __future__ import annotations

import base64
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from fsoc_pat.perception.adaptive_detector import AdaptiveDetector, AdaptiveDetectorConfig
from fsoc_pat.perception.detection_types import Candidate, Detection, DetectionStatus


@dataclass
class BenchmarkFrameRecord:
    """Telemetry and tracking record for a single processed video frame."""

    frame_idx: int
    timestamp_s: float
    detected: bool
    centroid_u: float | None = None
    centroid_v: float | None = None
    ground_truth_u: float | None = None
    ground_truth_v: float | None = None
    pixel_error: float | None = None
    cumulative_rmse: float = 0.0
    intensity: float = 0.0
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    status: str = "SEARCH"  # SEARCH, ACQUIRED, TRACKING, LOST


@dataclass
class VideoBenchmarkMetrics:
    """Summary metrics evaluated against PAT tracking specifications."""

    video_name: str = ""
    total_frames: int = 0
    processed_frames: int = 0
    duration_s: float = 0.0
    video_fps: float = 30.0
    processing_fps: float = 0.0

    # Tracking Benchmark Targets
    acquisition_time_s: float | None = None  # Spec: <= 2.0 sec
    reacquisition_time_s: float | None = None  # Spec: <= 1.0 sec
    mean_tracking_error_px: float = 0.0  # Spec: <= 10.0 px
    max_tracking_error_px: float = 0.0
    rmse_px: float = 0.0
    tracking_jitter_px: float = 0.0
    lock_retention_rate_pct: float = 0.0  # Spec: Target Loss < 5% -> Retention > 95%
    target_loss_pct: float = 100.0

    # Compliance booleans
    acquisition_compliant: bool = False
    tracking_error_compliant: bool = False
    retention_compliant: bool = False
    speed_compliant: bool = False


def resolve_video_file(video_path: str | Path) -> Path:
    """Resolve an input video path, searching common directories if bare filename provided."""
    p = Path(video_path)
    if p.is_file():
        return p.resolve()

    candidates = [
        Path.cwd() / p,
        Path.cwd() / "data" / p.name,
        Path.cwd() / "data" / "uploads" / p.name,
        Path.cwd() / "results" / p.name,
        Path.home() / "Downloads" / p.name,
        Path.home() / "Desktop" / p.name,
        Path.home() / p.name,
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()

    for search_root in [Path.cwd() / "data", Path.home() / "Downloads", Path.cwd()]:
        if search_root.is_dir():
            matches = list(search_root.glob(f"**/{p.name}"))
            if matches:
                return matches[0].resolve()

    raise FileNotFoundError(
        f"Video file not found: '{video_path}'. Searched in current directory, "
        f"data/uploads/, results/, and {Path.home() / 'Downloads'}."
    )


class VideoBenchmarkEngine:
    """Engine that ingests an MP4 video, bypasses PTZ camera, and tracks the beacon."""

    def __init__(self, detector_config: AdaptiveDetectorConfig | None = None) -> None:
        self.config = detector_config or AdaptiveDetectorConfig(
            min_snr=2.0,
            min_area_px=8.0,
            tophat_kernel_size=15,
            confidence_threshold=0.20,
        )
        self.detector = AdaptiveDetector(self.config)
        self._open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.reset()

    def reset(self) -> None:
        """Reset engine tracking state and metric accumulators."""
        self.detector.reset()
        self.records: list[BenchmarkFrameRecord] = []
        self.ground_truth_map: dict[int, tuple[float, float]] = {}

        self.is_locked = False
        self.consecutive_detections = 0
        self.consecutive_losses = 0

        self.acquisition_timestamp: float | None = None
        self.loss_timestamp: float | None = None
        self.reacq_durations: list[float] = []

        self.total_sq_error = 0.0
        self.evaluated_error_count = 0
        self.start_wall_time: float | None = None
        self.latest_frame_annotated: NDArray[np.uint8] | None = None

        # 4-state Constant Velocity Kalman tracking filter [u, dot_u, v, dot_v]
        self.kf_state: NDArray[np.float64] | None = None
        self.kf_cov: NDArray[np.float64] = np.eye(4, dtype=np.float64) * 100.0
        self.kf_Q: NDArray[np.float64] = np.eye(4, dtype=np.float64) * 0.1
        self.kf_Q[1, 1] = 5.0
        self.kf_Q[3, 3] = 5.0
        self.kf_H: NDArray[np.float64] = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float64)
        self.kf_R: NDArray[np.float64] = np.eye(2, dtype=np.float64) * 1.5

    def load_ground_truth_csv(self, csv_path: str | Path) -> None:
        """Load ground truth coordinates from a CSV file (frame_idx, gt_u, gt_v)."""
        self.ground_truth_map.clear()
        try:
            path = resolve_video_file(csv_path)
        except FileNotFoundError:
            path = Path(csv_path)

        if not path.is_file():
            return

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 3 and parts[0].isdigit():
                    idx = int(parts[0])
                    try:
                        u, v = float(parts[1]), float(parts[2])
                        self.ground_truth_map[idx] = (u, v)
                    except ValueError:
                        continue

    def process_frame(
        self, frame: NDArray[np.uint8], frame_idx: int, fps: float = 30.0
    ) -> BenchmarkFrameRecord:
        """Process a single frame from the video stream with optimal detection & tracking."""
        t_start = time.perf_counter()
        timestamp_s = frame_idx / max(fps, 1.0)

        # Convert to grayscale if BGR
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            disp_frame = frame.copy()
        else:
            gray = frame.copy()
            disp_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        H, W = gray.shape[:2]

        # 1. Spatial median filtering (suppresses impulse salt-and-pepper shot noise)
        gray_clean = cv2.medianBlur(gray, 3)

        # 2. Multi-stage beacon perception
        # For standard resolution (<= 640x480), evaluate primary AdaptiveDetector
        detected = False
        u_det, v_det = None, None
        intensity = 0.0
        confidence = 0.0

        if W <= 640 and H <= 480:
            detection: Detection = self.detector.detect(gray_clean)
            if detection.status == DetectionStatus.VALID and detection.best is not None:
                detected = True
                u_det = float(detection.best.u)
                v_det = float(detection.best.v)
                intensity = float(detection.best.peak_intensity)
                confidence = float(detection.confidence)

        if not detected:
            # High-speed dynamic adaptive thresholding + connected component analysis
            # Handles high-resolution Full-HD (1080p), large blooms, and high-noise backgrounds
            mean_v = float(np.mean(gray_clean))
            std_v = float(np.std(gray_clean))
            thresh_val = int(max(130.0, min(240.0, mean_v + 3.0 * std_v)))
            _, binary = cv2.threshold(gray_clean, thresh_val, 255, cv2.THRESH_BINARY)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._open_kernel)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )

            pred_pos = None
            if self.kf_state is not None:
                pred_pos = (float(self.kf_state[0]), float(self.kf_state[2]))

            best_score = -1.0
            best_cand = None

            for lab in range(1, num_labels):
                area = float(stats[lab, cv2.CC_STAT_AREA])
                if area < 3.0 or area > (W * H * 0.15):
                    continue

                cx, cy = float(centroids[lab, 0]), float(centroids[lab, 1])
                mask = (labels == lab)
                peak_val = float(np.max(gray_clean[mask]))

                dist_factor = 1.0
                if pred_pos is not None:
                    d = math.hypot(cx - pred_pos[0], cy - pred_pos[1])
                    dist_factor = math.exp(-0.5 * (d / 45.0)**2) if self.is_locked else 1.0 / (1.0 + d / 60.0)

                score = (peak_val - mean_v) * math.log(area + 1.0) * dist_factor
                if score > best_score:
                    best_score = score
                    # Sub-pixel intensity-weighted Center of Mass
                    y_idx, x_idx = np.where(mask)
                    weights = gray_clean[y_idx, x_idx].astype(np.float64) - mean_v
                    weights = np.maximum(weights, 0.0)
                    tot_w = float(np.sum(weights))
                    if tot_w > 0.0:
                        sub_u = float(np.sum(x_idx * weights) / tot_w)
                        sub_v = float(np.sum(y_idx * weights) / tot_w)
                    else:
                        sub_u, sub_v = cx, cy
                    best_cand = (sub_u, sub_v, area, peak_val)

            if best_cand is not None:
                detected = True
                u_det, v_det = best_cand[0], best_cand[1]
                intensity = best_cand[3]
                confidence = max(0.40, min(1.0, (intensity - mean_v) / max(1.0, 3.0 * std_v)))

        # 3. Kinematic Kalman State Tracker (Constant Velocity)
        dt = 1.0 / max(fps, 1.0)
        F = np.array([
            [1.0, dt,  0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, dt ],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

        if self.kf_state is None:
            if detected and u_det is not None and v_det is not None:
                self.kf_state = np.array([u_det, 0.0, v_det, 0.0], dtype=np.float64)
                self._prev_det = (u_det, v_det, timestamp_s)
            else:
                self.kf_state = np.array([W / 2.0, 0.0, H / 2.0, 0.0], dtype=np.float64)
            pred_pos = (float(self.kf_state[0]), float(self.kf_state[2]))
            pred_cov = self.kf_cov.copy()
        else:
            if getattr(self, "_prev_det", None) is not None and detected and u_det is not None and v_det is not None:
                dt_step = max(dt, 1e-3)
                vx = (u_det - self._prev_det[0]) / dt_step
                vy = (v_det - self._prev_det[1]) / dt_step
                self.kf_state[1] = vx
                self.kf_state[3] = vy
                self._prev_det = None

            pred_state = F @ self.kf_state
            pred_cov = F @ self.kf_cov @ F.T + self.kf_Q
            pred_pos = (float(pred_state[0]), float(pred_state[2]))

            if detected and u_det is not None and v_det is not None:
                meas = np.array([u_det, v_det], dtype=np.float64)
                y = meas - np.array([pred_pos[0], pred_pos[1]])
                S = self.kf_H @ pred_cov @ self.kf_H.T + self.kf_R
                K = pred_cov @ self.kf_H.T @ np.linalg.inv(S)
                self.kf_state = pred_state + K @ y
                self.kf_cov = (np.eye(4, dtype=np.float64) - K @ self.kf_H) @ pred_cov
            else:
                self.kf_state = pred_state
                self.kf_cov = pred_cov

        t_proc_ms = (time.perf_counter() - t_start) * 1000.0

        # 4. Ground truth evaluation:
        # If external ground truth CSV provided: compare measured centroid against GT
        # If no CSV provided: compare measured centroid against Kalman predicted track
        has_gt = frame_idx in self.ground_truth_map
        if has_gt:
            gt_u, gt_v = self.ground_truth_map[frame_idx]
        else:
            gt_u, gt_v = pred_pos[0], pred_pos[1]

        pixel_error = None
        if detected and u_det is not None and v_det is not None:
            if has_gt:
                pixel_error = math.hypot(u_det - gt_u, v_det - gt_v)
            else:
                pixel_error = math.hypot(u_det - pred_pos[0], v_det - pred_pos[1])

            self.total_sq_error += pixel_error * pixel_error
            self.evaluated_error_count += 1
            self.consecutive_detections += 1
            self.consecutive_losses = 0

            # State tracking: Acquisition
            if not self.is_locked and self.consecutive_detections >= 3:
                self.is_locked = True
                if self.acquisition_timestamp is None:
                    self.acquisition_timestamp = timestamp_s

                # Check if this was a re-acquisition after loss
                if self.loss_timestamp is not None:
                    reacq_dt = timestamp_s - self.loss_timestamp
                    if reacq_dt > 0:
                        self.reacq_durations.append(reacq_dt)
                    self.loss_timestamp = None

            status = "TRACKING" if self.is_locked else "ACQUIRING"
        else:
            self.consecutive_losses += 1
            self.consecutive_detections = 0
            if self.is_locked and self.consecutive_losses >= 3:
                self.is_locked = False
                self.loss_timestamp = timestamp_s
            status = "LOST" if self.is_locked else "SEARCH"

        # Calculate running RMSE
        rmse = 0.0
        if self.evaluated_error_count > 0:
            rmse = math.sqrt(self.total_sq_error / self.evaluated_error_count)

        # Annotate frame with HUD and crosshairs
        self._annotate_hud(disp_frame, u_det, v_det, gt_u, gt_v, status, pixel_error, rmse, t_proc_ms)
        self.latest_frame_annotated = disp_frame

        rec = BenchmarkFrameRecord(
            frame_idx=frame_idx,
            timestamp_s=round(timestamp_s, 3),
            detected=detected,
            centroid_u=round(u_det, 2) if u_det is not None else None,
            centroid_v=round(v_det, 2) if v_det is not None else None,
            ground_truth_u=round(gt_u, 2),
            ground_truth_v=round(gt_v, 2),
            pixel_error=round(pixel_error, 2) if pixel_error is not None else None,
            cumulative_rmse=round(rmse, 2),
            intensity=round(intensity, 1),
            confidence=round(confidence, 3),
            processing_time_ms=round(t_proc_ms, 2),
            status=status,
        )
        self.records.append(rec)
        return rec

    def _annotate_hud(
        self,
        img: NDArray[np.uint8],
        u: float | None,
        v: float | None,
        gt_u: float,
        gt_v: float,
        status: str,
        err_px: float | None,
        rmse_px: float,
        proc_ms: float,
    ) -> None:
        """Render optical tracking HUD overlays onto frame."""
        H, W = img.shape[:2]

        # Draw Ground Truth Crosshair (Cyan)
        cv2.drawMarker(
            img, (int(round(gt_u)), int(round(gt_v))),
            (255, 255, 0), cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA,
        )

        # Draw Detected Centroid (Green if locked, Red/Orange if searching)
        if u is not None and v is not None:
            iu, iv = int(round(u)), int(round(v))
            color = (0, 255, 0) if status == "TRACKING" else (0, 165, 255)
            # Center target reticle
            cv2.circle(img, (iu, iv), 12, color, 1, cv2.LINE_AA)
            cv2.circle(img, (iu, iv), 2, (0, 0, 255), -1)
            # Bounding box
            cv2.rectangle(img, (iu - 14, iv - 14), (iu + 14, iv + 14), color, 1)
            # Line connecting centroid to ground truth
            cv2.line(img, (iu, iv), (int(round(gt_u)), int(round(gt_v))), (0, 200, 255), 1, cv2.LINE_AA)

        # Top HUD Status Banner
        status_color = (0, 255, 0) if status == "TRACKING" else (0, 140, 255) if status == "ACQUIRING" else (0, 0, 255)
        cv2.putText(img, f"PTZ-BYPASS STREAM | STATUS: {status}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

        # Metric Badges
        err_str = f"Error: {err_px:.1f}px" if err_px is not None else "Error: N/A"
        rmse_str = f"RMSE: {rmse_px:.1f}px"
        fps_str = f"Proc: {proc_ms:.1f}ms ({1000.0/max(proc_ms, 0.1):.0f} FPS)"

        cv2.putText(img, f"{err_str}  |  {rmse_str}  |  {fps_str}", (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1, cv2.LINE_AA)

    def process_video_file(
        self, video_path: str | Path, ground_truth_csv: str | Path | None = None
    ) -> VideoBenchmarkMetrics:
        """Process an entire MP4 video file from start to finish."""
        path = resolve_video_file(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.reset()

        if ground_truth_csv:
            self.load_ground_truth_csv(ground_truth_csv)
        elif path.with_suffix(".csv").is_file():
            self.load_ground_truth_csv(path.with_suffix(".csv"))

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        t_wall_start = time.perf_counter()

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            self.process_frame(frame, frame_idx, fps)
            frame_idx += 1

        cap.release()
        wall_time = time.perf_counter() - t_wall_start
        metrics = self.compute_metrics(total_frames, fps, wall_time)
        metrics.video_name = path.name
        return metrics

    def compute_metrics(
        self, total_frames: int, video_fps: float, wall_time_s: float
    ) -> VideoBenchmarkMetrics:
        """Compute aggregate benchmark metrics compliant with PAT tracking specifications."""
        processed = len(self.records)
        duration_s = processed / max(video_fps, 1.0)
        proc_fps = processed / max(wall_time_s, 0.001)

        valid_errors = [r.pixel_error for r in self.records if r.pixel_error is not None]
        mean_err = float(np.mean(valid_errors)) if valid_errors else 0.0
        max_err = float(np.max(valid_errors)) if valid_errors else 0.0
        std_err = float(np.std(valid_errors)) if valid_errors else 0.0
        rmse = math.sqrt(self.total_sq_error / max(self.evaluated_error_count, 1))

        # Lock retention
        locked_frames = sum(1 for r in self.records if r.status in ("TRACKING", "ACQUIRING"))
        lock_rate = (locked_frames / max(processed, 1)) * 100.0
        loss_rate = 100.0 - lock_rate

        # Re-acquisition time (average duration of signal recovery)
        mean_reacq = float(np.mean(self.reacq_durations)) if self.reacq_durations else 0.0

        acq_time = self.acquisition_timestamp if self.acquisition_timestamp is not None else duration_s

        metrics = VideoBenchmarkMetrics(
            total_frames=total_frames,
            processed_frames=processed,
            duration_s=round(duration_s, 2),
            video_fps=round(video_fps, 1),
            processing_fps=round(proc_fps, 1),
            acquisition_time_s=round(acq_time, 2) if acq_time is not None else None,
            reacquisition_time_s=round(mean_reacq, 2),
            mean_tracking_error_px=round(mean_err, 2),
            max_tracking_error_px=round(max_err, 2),
            rmse_px=round(rmse, 2),
            tracking_jitter_px=round(std_err, 2),
            lock_retention_rate_pct=round(lock_rate, 1),
            target_loss_pct=round(loss_rate, 1),
            # ISRO compliance criteria
            acquisition_compliant=(acq_time <= 2.0) if acq_time is not None else False,
            tracking_error_compliant=(mean_err <= 10.0),
            retention_compliant=(loss_rate < 5.0),
            speed_compliant=(proc_fps >= 20.0),
        )
        return metrics

    def export_csv(self, output_path: str | Path) -> None:
        """Export frame-by-frame tracking records to CSV."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "frame_idx,timestamp_s,detected,centroid_u,centroid_v,"
                "ground_truth_u,ground_truth_v,pixel_error,cumulative_rmse,"
                "intensity,confidence,processing_time_ms,status\n"
            )
            for r in self.records:
                f.write(
                    f"{r.frame_idx},{r.timestamp_s},{r.detected},"
                    f"{r.centroid_u if r.centroid_u is not None else ''},"
                    f"{r.centroid_v if r.centroid_v is not None else ''},"
                    f"{r.ground_truth_u},{r.ground_truth_v},"
                    f"{r.pixel_error if r.pixel_error is not None else ''},"
                    f"{r.cumulative_rmse},{r.intensity},{r.confidence},"
                    f"{r.processing_time_ms},{r.status}\n"
                )

    def get_latest_jpeg_base64(self) -> str:
        """Return the latest annotated frame as a Base64-encoded JPEG image."""
        if self.latest_frame_annotated is None:
            return ""
        success, buf = cv2.imencode(".jpg", self.latest_frame_annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not success:
            return ""
        return base64.b64encode(buf).decode("utf-8")


class BenchmarkVideoGenerator:
    """Generates synthetic compliant MP4 video benchmarks."""

    @staticmethod
    def generate_benchmark_video(
        output_mp4: str | Path,
        duration_s: float = 10.0,
        fps: float = 30.0,
        resolution: tuple[int, int] = (640, 480),
        virtual_space_size: tuple[int, int] | None = None,
        motion_type: str = "figure_of_8",
        target_shape: str = "square",
        target_size: int = 10,
        pan_tilt_speed_limit_deg_s: float = 5.0,
        preset: str = "custom",
        contrast_reduction: float = 0.0,
        brightness_reduction: float = 0.0,
        camera_jitter_px: float = 0.0,
        platform_motion_px: float = 0.0,
        enable_salt_pepper: bool = True,
        salt_pepper_ratio: float = 0.10,
        enable_gaussian: bool = True,
        gaussian_std: float = 12.0,
        enable_poisson: bool = True,
        seed: int = 42,
    ) -> tuple[Path, Path]:
        """Generate an MP4 benchmark video and accompanying ground truth CSV.

        Supports:
        - 2000 x 2000 Virtual Screen Space with 640 x 480 Viewport
        - Square (10 x 10 px) default target preset
        - Camera Pan & Tilt Speed Limiter (5.0 - 10.0 °/s, default 5.0 °/s)
        - Explicit "Haze" and "Low Light" disturbance presets
        - Contrast & Brightness Reduction Sliders
        - Camera Jitter Control (±20 px/frame)
        - Platform Motion Injection (±20 px/frame)

        Returns:
            (mp4_path, ground_truth_csv_path)
        """
        mp4_path = Path(output_mp4)
        mp4_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path = mp4_path.with_suffix(".csv")

        W, H = resolution
        if virtual_space_size is None:
            VW, VH = (2000, 2000) if resolution == (640, 480) else resolution
        else:
            VW, VH = virtual_space_size

        total_frames = int(round(duration_s * fps))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(mp4_path), fourcc, fps, (W, H), isColor=True)

        rng = np.random.default_rng(seed)
        gt_records: list[str] = ["frame_idx,gt_u,gt_v,is_visible\n"]

        # Handle disturbance presets
        p_lower = (preset or "custom").strip().lower()
        base_bg = 15.0
        if p_lower == "haze":
            contrast_reduction = max(contrast_reduction, 0.45)
            gaussian_std = max(gaussian_std, 10.0)
            base_bg = 55.0
        elif p_lower in ("low_light", "low light"):
            brightness_reduction = max(brightness_reduction, 0.60)
            base_bg = 8.0
            enable_poisson = True

        # Virtual space trajectory bounds
        cx_v, cy_v = VW / 2.0, VH / 2.0
        rx_v, ry_v = VW * 0.35, VH * 0.35

        # Pan/tilt speed limiter in pixels per second
        # 2000 px maps across 90° field of regard (~22.22 px/deg)
        px_per_deg = VW / 90.0
        speed_limit_deg = max(5.0, min(10.0, float(pan_tilt_speed_limit_deg_s)))
        max_px_per_sec = speed_limit_deg * px_per_deg
        max_step_per_frame = max_px_per_sec / fps

        # Initial camera viewport position in virtual space
        cam_u, cam_v = cx_v, cy_v

        for i in range(total_frames):
            t = i / fps
            dt = 1.0 / fps

            # 1. Target kinematics in virtual screen space (2000 x 2000)
            if motion_type == "figure_of_8":
                omega = 2.0 * math.pi / max(duration_s / 2.0, 1.0)
                u_virt = cx_v + rx_v * math.sin(omega * t)
                v_virt = cy_v + 0.5 * ry_v * math.sin(2.0 * omega * t)
            elif motion_type == "circular":
                omega = 2.0 * math.pi / max(duration_s / 2.0, 1.0)
                u_virt = cx_v + rx_v * 0.7 * math.cos(omega * t)
                v_virt = cy_v + ry_v * 0.7 * math.sin(omega * t)
            else:  # linear
                speed_x = (VW * 0.6) / duration_s
                u_virt = (VW * 0.2) + (speed_x * t) % (VW * 0.6)
                v_virt = cy_v + (VH * 0.2) * math.sin(1.5 * t)

            # 2. Camera Viewport Tracking with Pan & Tilt Speed Limiter
            if VW > W or VH > H:
                # Slew camera towards target in virtual space
                du_req = u_virt - cam_u
                dv_req = v_virt - cam_v
                dist_req = math.hypot(du_req, dv_req)
                if dist_req > 1e-4:
                    step = min(dist_req, max_step_per_frame)
                    cam_u += (du_req / dist_req) * step
                    cam_v += (dv_req / dist_req) * step
            else:
                cam_u, cam_v = cx_v, cy_v

            # 3. Camera Jitter (±20 px/frame)
            j_u = float(rng.uniform(-camera_jitter_px, camera_jitter_px)) if camera_jitter_px > 0 else 0.0
            j_v = float(rng.uniform(-camera_jitter_px, camera_jitter_px)) if camera_jitter_px > 0 else 0.0

            # 4. Platform Motion Injection (±20 px/frame)
            if platform_motion_px > 0:
                p_u = float(platform_motion_px * math.sin(2.0 * math.pi * 0.7 * t) + rng.normal(0, 0.15 * platform_motion_px))
                p_v = float(platform_motion_px * math.cos(2.0 * math.pi * 0.5 * t) + rng.normal(0, 0.15 * platform_motion_px))
                p_u = max(-platform_motion_px, min(platform_motion_px, p_u))
                p_v = max(-platform_motion_px, min(platform_motion_px, p_v))
            else:
                p_u, p_v = 0.0, 0.0

            # 5. Transform from 2000x2000 virtual space to 640x480 viewport
            if VW > W or VH > H:
                u_vp = (u_virt - cam_u) + (W / 2.0) + j_u + p_u
                v_vp = (v_virt - cam_v) + (H / 2.0) + j_v + p_v
            else:
                u_vp = u_virt + j_u + p_u
                v_vp = v_virt + j_v + p_v

            is_visible = int(0 <= u_vp < W and 0 <= v_vp < H)

            # 6. Render frame background
            frame = rng.normal(base_bg, 4.0, (H, W)).astype(np.float64)

            # 7. Render target spot (Default: Square 10x10 px)
            if is_visible:
                half = target_size / 2.0
                if target_shape.lower() in ("square", "box"):
                    x1 = max(0, int(round(u_vp - half)))
                    y1 = max(0, int(round(v_vp - half)))
                    x2 = min(W, int(round(u_vp + half)))
                    y2 = min(H, int(round(v_vp + half)))
                    if x2 > x1 and y2 > y1:
                        frame[y1:y2, x1:x2] += 220.0
                else:  # gaussian circle
                    sigma = max(1.0, target_size / 2.5)
                    y_min = max(0, int(v_vp - 3 * sigma))
                    y_max = min(H, int(v_vp + 3 * sigma + 1))
                    x_min = max(0, int(u_vp - 3 * sigma))
                    x_max = min(W, int(u_vp + 3 * sigma + 1))
                    if x_max > x_min and y_max > y_min:
                        ys, xs = np.mgrid[y_min:y_max, x_min:x_max]
                        dist_sq = (xs - u_vp) ** 2 + (ys - v_vp) ** 2
                        frame[y_min:y_max, x_min:x_max] += 240.0 * np.exp(-dist_sq / (2 * sigma * sigma))

            # 8. Sensor noise & disturbances
            if enable_gaussian and gaussian_std > 0:
                frame += rng.normal(0, gaussian_std, (H, W))

            if enable_poisson:
                clipped = np.clip(frame, 0, 255)
                noisy = rng.poisson(np.maximum(clipped * 0.5, 0))
                frame = clipped + (noisy - clipped * 0.5) * 0.5

            if enable_salt_pepper and salt_pepper_ratio > 0:
                prob = min(max(salt_pepper_ratio, 0.0), 1.0)
                rand_mat = rng.uniform(0.0, 1.0, (H, W))
                frame[rand_mat < (prob / 2.0)] = 0.0
                frame[(rand_mat >= (prob / 2.0)) & (rand_mat < prob)] = 255.0

            # 9. Contrast & Brightness Reduction Sliders
            if contrast_reduction > 0.0:
                c_scale = 1.0 - min(1.0, max(0.0, float(contrast_reduction)))
                mean_val = np.mean(frame)
                frame = mean_val + (frame - mean_val) * c_scale

            if brightness_reduction > 0.0:
                b_scale = 1.0 - min(1.0, max(0.0, float(brightness_reduction)))
                frame = frame * b_scale

            frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
            bgr_frame = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR)
            writer.write(bgr_frame)

            gt_records.append(f"{i},{u_vp:.2f},{v_vp:.2f},{is_visible}\n")

        writer.release()

        # Write Ground Truth CSV
        with open(csv_path, "w", encoding="utf-8") as f:
            f.writelines(gt_records)

        return mp4_path, csv_path
