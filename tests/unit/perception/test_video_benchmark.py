"""Unit tests for VideoBenchmarkEngine and BenchmarkVideoGenerator (ISRO Benchmark-2)."""

import math
from pathlib import Path

import numpy as np
import pytest

from fsoc_pat.perception.video_benchmark import (
    BenchmarkVideoGenerator,
    VideoBenchmarkEngine,
    VideoBenchmarkMetrics,
)


@pytest.fixture
def sample_benchmark_video(tmp_path: Path) -> tuple[Path, Path]:
    """Generate a quick 2-second 30 FPS benchmark video with Figure-of-8 motion."""
    video_path = tmp_path / "test_benchmark.mp4"
    return BenchmarkVideoGenerator.generate_benchmark_video(
        output_mp4=video_path,
        duration_s=2.0,
        fps=30.0,
        resolution=(320, 240),
        motion_type="figure_of_8",
        target_shape="square",
        target_size=10,
        enable_salt_pepper=True,
        salt_pepper_ratio=0.08,
        enable_gaussian=True,
        gaussian_std=8.0,
        enable_poisson=True,
        seed=123,
    )


def test_video_generator_outputs_valid_files(sample_benchmark_video: tuple[Path, Path]) -> None:
    """BenchmarkVideoGenerator should output non-empty mp4 and csv files."""
    mp4_path, csv_path = sample_benchmark_video
    assert mp4_path.is_file()
    assert csv_path.is_file()
    assert mp4_path.stat().st_size > 1000
    assert csv_path.stat().st_size > 100


def test_video_benchmark_engine_processing(sample_benchmark_video: tuple[Path, Path], tmp_path: Path) -> None:
    """VideoBenchmarkEngine should ingest MP4, compute metrics, and achieve ISRO compliance."""
    mp4_path, csv_path = sample_benchmark_video
    engine = VideoBenchmarkEngine()

    metrics: VideoBenchmarkMetrics = engine.process_video_file(mp4_path, ground_truth_csv=csv_path)

    # 2 seconds @ 30 FPS = 60 frames
    assert metrics.processed_frames == 60
    assert metrics.total_frames == 60
    assert metrics.processing_fps > 20.0
    assert metrics.speed_compliant is True

    # Error should be well within ISRO <= 10 pixel threshold
    assert metrics.mean_tracking_error_px <= 10.0
    assert metrics.tracking_error_compliant is True
    assert metrics.rmse_px <= 10.0

    # Acquisition time <= 2.0s
    assert metrics.acquisition_time_s is not None
    assert metrics.acquisition_time_s <= 2.0
    assert metrics.acquisition_compliant is True

    # Lock retention > 95%
    assert metrics.lock_retention_rate_pct >= 90.0

    # Base64 JPEG frame overlay check
    b64_img = engine.get_latest_jpeg_base64()
    assert len(b64_img) > 100

    # Export CSV check
    export_csv = tmp_path / "benchmark_report.csv"
    engine.export_csv(export_csv)
    assert export_csv.is_file()
    assert export_csv.stat().st_size > 500


def test_video_benchmark_with_virtual_space_and_disturbances(tmp_path: Path) -> None:
    """Test 2000x2000 virtual screen space, square target, speed limiter, jitter, and disturbances."""
    video_path = tmp_path / "spec_discrepancies_benchmark.mp4"
    mp4_path, csv_path = BenchmarkVideoGenerator.generate_benchmark_video(
        output_mp4=video_path,
        duration_s=1.0,
        fps=30.0,
        resolution=(640, 480),
        virtual_space_size=(2000, 2000),
        motion_type="circular",
        target_shape="square",
        target_size=10,
        pan_tilt_speed_limit_deg_s=5.0,
        preset="haze",
        contrast_reduction=0.25,
        brightness_reduction=0.15,
        camera_jitter_px=10.0,
        platform_motion_px=8.0,
        seed=42,
    )
    assert mp4_path.is_file()
    assert csv_path.is_file()

    # Verify ground truth CSV was generated
    with open(csv_path) as f:
        lines = f.readlines()
    assert len(lines) == 31  # header + 30 frames
    header = lines[0].strip()
    assert header == "frame_idx,gt_u,gt_v,is_visible"

    # Test engine processing
    engine = VideoBenchmarkEngine()
    metrics = engine.process_video_file(mp4_path, ground_truth_csv=csv_path)
    assert metrics.processed_frames == 30
    assert metrics.total_frames == 30

