# Changelog

All notable changes to OptiLign will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-06

### Added
- Phase 1: Mathematical Core + Ground Truth
  - Core domain types (Position3D, Velocity3D, AngularError, SimulationTime, etc.)
  - Simulation clock with real-time, accelerated, and offline modes
  - State objects (CameraState, TargetState, GroundTruthState)
  - Protocol interfaces (Trajectory, Detector, Estimator, Controller)
  - Event bus for decoupled subsystem communication
  - Coordinate frame system (World, Platform, Gimbal, Camera)
  - Rotation matrices and Euler angle conversions
  - Quaternion algebra
  - Coordinate frame transformations
  - Camera intrinsics model (FOV ↔ focal length)
  - 3D-to-pixel projection and pixel-to-ray inverse
  - Line-of-sight geometry
  - Angular pointing error computation
  - Trajectory library (stationary, linear, circular, sinusoidal, spiral, maneuver, random)
  - Target and beacon property models
  - Gimbal dynamics with rate limiting, lag, deadband, noise
  - Virtual camera abstraction
  - Simulation world and engine (headless operation)
  - Configuration system with YAML loading and validation
  - Default and nominal scenario configurations
  - Comprehensive unit and integration tests
  - Architecture documentation
