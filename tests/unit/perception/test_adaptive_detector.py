"""Unit tests for environment-adaptive optical detector."""

import cv2
import numpy as np
import pytest

from fsoc_pat.perception.adaptive_detector import AdaptiveDetector, AdaptiveDetectorConfig


def test_adaptive_detector_finds_beacon_in_noisy_background():
    detector = AdaptiveDetector()
    detector.reset()

    # Synthetic image (480x640) with background clutter and a diffraction beacon spot
    img = np.random.normal(30.0, 5.0, (480, 640)).astype(np.uint8)

    # Add background illumination gradient (e.g. city light wash / dawn sky)
    y, x = np.mgrid[0:480, 0:640]
    gradient = (x / 640.0 * 50.0).astype(np.uint8)
    img = cv2.add(img, gradient)

    # Inject beacon at (320, 240)
    beacon_spot = np.exp(-((x - 320.0)**2 + (y - 240.0)**2) / (2.0 * 3.0**2)) * 200.0
    img = cv2.add(img, beacon_spot.astype(np.uint8))

    det = detector.detect(img)
    assert det.detected is True
    assert det.best is not None
    assert abs(det.best.u - 320.0) < 1.5
    assert abs(det.best.v - 240.0) < 1.5
    assert det.confidence > 0.35


def test_adaptive_detector_rejects_empty_and_noise():
    detector = AdaptiveDetector()
    detector.reset()

    # Dark noisy image without any beacon
    img = np.random.normal(15.0, 3.0, (480, 640)).astype(np.uint8)
    det = detector.detect(img)

    assert det.detected is False
