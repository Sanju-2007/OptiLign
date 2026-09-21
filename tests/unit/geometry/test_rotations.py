"""Tests for rotation matrices and Euler angle conversion."""

import math

import numpy as np
import pytest

from fsoc_pat.geometry.rotations import (
    euler_to_rotation_matrix,
    is_valid_rotation_matrix,
    rotation_matrix_to_euler,
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
)


class TestElementaryRotations:
    """Tests for elementary rotation matrices."""

    def test_identity_rotation_x(self) -> None:
        R = rotation_matrix_x(0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-15)

    def test_identity_rotation_y(self) -> None:
        R = rotation_matrix_y(0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-15)

    def test_identity_rotation_z(self) -> None:
        R = rotation_matrix_z(0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-15)

    def test_90deg_rotation_z(self) -> None:
        """Rz(π/2) should map [1,0,0] → [0,1,0]."""
        R = rotation_matrix_z(math.pi / 2)
        v = np.array([1.0, 0.0, 0.0])
        result = R @ v
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-10)

    def test_90deg_rotation_x(self) -> None:
        """Rx(π/2) should map [0,1,0] → [0,0,1]."""
        R = rotation_matrix_x(math.pi / 2)
        v = np.array([0.0, 1.0, 0.0])
        result = R @ v
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0], atol=1e-10)

    def test_90deg_rotation_y(self) -> None:
        """Ry(π/2) should map [0,0,1] → [1,0,0]... but let's check
        Ry(π/2) maps [1,0,0] → [0,0,-1]."""
        R = rotation_matrix_y(math.pi / 2)
        v = np.array([1.0, 0.0, 0.0])
        result = R @ v
        np.testing.assert_allclose(result, [0.0, 0.0, -1.0], atol=1e-10)

    def test_rotation_orthogonal(self) -> None:
        """Every rotation matrix must satisfy R^T R = I."""
        for angle in [0.1, 0.5, 1.0, math.pi / 4, math.pi / 2, math.pi]:
            for rot_fn in [rotation_matrix_x, rotation_matrix_y, rotation_matrix_z]:
                R = rot_fn(angle)
                np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)

    def test_rotation_det_one(self) -> None:
        """Rotation matrix determinant must be +1."""
        for angle in [0.3, 1.2, -0.7]:
            for rot_fn in [rotation_matrix_x, rotation_matrix_y, rotation_matrix_z]:
                R = rot_fn(angle)
                assert abs(np.linalg.det(R) - 1.0) < 1e-12

    def test_composition_associativity(self) -> None:
        """(Rx · Ry) · Rz = Rx · (Ry · Rz)."""
        Rx = rotation_matrix_x(0.3)
        Ry = rotation_matrix_y(0.5)
        Rz = rotation_matrix_z(0.7)
        np.testing.assert_allclose((Rx @ Ry) @ Rz, Rx @ (Ry @ Rz), atol=1e-12)

    def test_inverse_is_transpose(self) -> None:
        R = rotation_matrix_z(1.0)
        R_inv = rotation_matrix_z(-1.0)
        np.testing.assert_allclose(R.T, R_inv, atol=1e-12)


class TestEulerConversion:
    """Tests for Euler ↔ rotation matrix conversion."""

    def test_identity_euler(self) -> None:
        R = euler_to_rotation_matrix(0.0, 0.0, 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-15)

    def test_round_trip(self) -> None:
        """euler → matrix → euler should recover original angles."""
        test_cases = [
            (0.1, 0.2, 0.3),
            (-0.5, 0.0, 1.0),
            (0.0, 0.0, math.pi / 4),
            (0.3, -0.3, 0.7),
        ]
        for roll, pitch, yaw in test_cases:
            R = euler_to_rotation_matrix(roll, pitch, yaw)
            r2, p2, y2 = rotation_matrix_to_euler(R)
            # Verify the reconstructed matrix matches
            R2 = euler_to_rotation_matrix(r2, p2, y2)
            np.testing.assert_allclose(R, R2, atol=1e-10)

    def test_is_valid_rotation_matrix(self) -> None:
        R = euler_to_rotation_matrix(0.5, 0.3, 0.7)
        assert is_valid_rotation_matrix(R)

    def test_invalid_rotation_matrix(self) -> None:
        assert not is_valid_rotation_matrix(np.eye(4))
        assert not is_valid_rotation_matrix(np.zeros((3, 3)))
        assert not is_valid_rotation_matrix(2 * np.eye(3))
