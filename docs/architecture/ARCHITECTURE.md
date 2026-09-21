# OptiLign — System Architecture

## Overview

OptiLign implements a closed-loop virtual coarse Pointing, Acquisition and Tracking
(PAT) system for Free-Space Optical Communication (FSOC). The architecture enforces
strict separation between the simulation core and the user interface.

## Coordinate Convention

| Axis | Direction | Description |
|------|-----------|-------------|
| X | Forward | Optical axis / boresight |
| Y | Right | Horizontal |
| Z | Down | Vertical |

- **Handedness**: Right-handed
- **Rotation convention**: ZYX intrinsic Euler angles (yaw → pitch → roll)
- **Angle units**: Radians internally; degrees for display only
- **SI units**: meters, m/s, m/s², radians, rad/s, seconds

## Layered Architecture

```
┌──────────────────────────────────────────────────┐
│                    GUI (ui/)                      │  ← Phase 7
│  PySide6 widgets, viewmodels, controllers        │
└──────────────────┬───────────────────────────────┘
                   │ calls
┌──────────────────▼───────────────────────────────┐
│            Application Services                   │  ← Phase 7
│  simulation_service, experiment_service           │
└──────────────────┬───────────────────────────────┘
                   │ orchestrates
┌──────────────────▼───────────────────────────────┐
│            Experiment Engine                      │  ← Phase 5-7
│  scenario, runner, benchmark, monte_carlo         │
└──────────────────┬───────────────────────────────┘
                   │ runs
┌──────────────────▼───────────────────────────────┐
│         Simulation Engine (simulation/)           │  ← Phase 1
│  clock, world, targets, camera, gimbal, engine    │
│                                                   │
│  ┌─────────┐  ┌──────────┐  ┌─────────────────┐ │
│  │Perception│→│Estimation │→│    Control       │ │  ← Phase 2-4
│  │(percept/)│ │(estim/)   │ │   (control/)     │ │
│  └─────────┘  └──────────┘  └────────┬────────┘ │
│                                       │          │
│  ┌──────────┐  ┌──────────┐  ┌───────▼────────┐ │
│  │Disturb.  │  │PAT State │  │  Gimbal Dyn.   │ │  ← Phase 5
│  │(disturb/)│  │(pat/)    │  │  (simulation/) │ │
│  └──────────┘  └──────────┘  └────────────────┘ │
└──────────────────────────────────────────────────┘
         │                      │
┌────────▼──────────┐  ┌───────▼─────────────────┐
│ Geometry (geom/)  │  │   FSOC (fsoc/)          │  ← Phase 1, 6
│ frames, rotations │  │ beam, alignment, link   │
│ projection, LOS   │  │                         │
│ angular error     │  │                         │
└───────────────────┘  └─────────────────────────┘
         │
┌────────▼──────────┐
│   Core (core/)    │  ← Phase 1
│ types, clock,     │
│ interfaces, events│
│ state, errors     │
└───────────────────┘
```

## Dependency Rules

1. **Core** depends on nothing (only numpy).
2. **Geometry** depends on Core only.
3. **Simulation** depends on Core + Geometry.
4. **Perception/Estimation/Control** depend on Core + Geometry.
5. **PAT/Disturbances** depend on Core.
6. **FSOC** depends on Core + Geometry.
7. **Experiments/Metrics** depend on everything above.
8. **Application** depends on Experiments/Metrics.
9. **UI** depends on Application. **UI never contains algorithms.**

## Key Protocols (Plugin Interfaces)

| Protocol | Purpose | Module |
|----------|---------|--------|
| `TrajectoryProtocol` | Target motion models | `core/interfaces.py` |
| `DetectorProtocol` | Beacon detection | `core/interfaces.py` |
| `EstimatorProtocol` | State estimation (KF/EKF/UKF) | `core/interfaces.py` |
| `ControllerProtocol` | Pan/tilt control (PID/LQR/MPC) | `core/interfaces.py` |
| `DisturbanceProtocol` | Disturbance injection | `core/interfaces.py` |

New algorithms implement these protocols and are registered via configuration.
No modification to the engine, GUI, or metrics code is needed.

## Frame Hierarchy

```
WORLD (fixed inertial)
  └── PLATFORM (camera body)
        └── GIMBAL (pan-tilt mechanism)
              └── CAMERA (sensor frame)
```

Pan = rotation about PLATFORM Z-axis.
Tilt = rotation about resulting Y-axis.

## Simulation Loop (Phase 1)

```
CLOCK.tick()
  → target.state_at(t)         # ground truth position
  → camera.get_state_at(t)     # current camera state
  → project_point_to_pixel()   # where target appears on image
  → angular_error()            # how far off boresight
  → record_ground_truth()      # log everything
  → NEXT STEP
```

## Ground Truth Contract

Ground truth is **ONLY** used for:
- Simulation truth (generating the world)
- Validation (comparing tracking performance)
- Metrics (quantitative evaluation)
- Visualization overlays (when explicitly enabled)

Ground truth is **NEVER** fed to:
- Detector
- Estimator
- Controller
- PAT state machine
