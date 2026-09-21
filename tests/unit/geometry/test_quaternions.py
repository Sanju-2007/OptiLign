"""Tests for quaternion algebra."""

import math

import numpy as np
import pytest

from fsoc_pat.geometry.quaternions import Quaternion, slerp


class TestQuaternion:
    """Unit tests for Quaternion operations."""

    def test_identity(self) -> None:
        q = Quaternion.identity()
        assert q.w == 1.0
        assert q.x == 0.0
        assert q.y == 0.0
        assert q.z == 0.0

    def test_norm(self) -> None:
        q = Quaternion(1.0, 0.0, 0.0, 0.0)
        assert q.norm == pytest.approx(1.0)

    def test_normalization(self) -> None:
        q = Quaternion(2.0, 0.0, 0.0, 0.0)
        qn = q.normalized()
        assert qn.norm == pytest.approx(1.0)
        assert qn.w == pytest.approx(1.0)

    def test_conjugate(self) -> None:
        q = Quaternion(1.0, 2.0, 3.0, 4.0).normalized()
        qc = q.conjugate()
        assert qc.w == pytest.approx(q.w)
        assert qc.x == pytest.approx(-q.x)
        assert qc.y == pytest.approx(-q.y)
        assert qc.z == pytest.approx(-q.z)

    def test_multiply_identity(self) -> None:
        q = Quaternion.from_euler(0.3, 0.5, 0.7)
        identity = Quaternion.identity()
        result = q.multiply(identity)
        assert result.w == pytest.approx(q.w, abs=1e-10)
        assert result.x == pytest.approx(q.x, abs=1e-10)

    def test_multiply_conjugate_is_identity(self) -> None:
        q = Quaternion.from_euler(0.3, 0.5, 0.7)
        result = q.multiply(q.conjugate())
        assert result.w == pytest.approx(1.0, abs=1e-10)
        assert abs(result.x) < 1e-10
        assert abs(result.y) < 1e-10
        assert abs(result.z) < 1e-10

    def test_euler_round_trip(self) -> None:
        """from_euler → to_euler should recover original angles."""
        roll, pitch, yaw = 0.3, 0.2, 0.5
        q = Quaternion.from_euler(roll, pitch, yaw)
        r2, p2, y2 = q.to_euler()
        assert r2 == pytest.approx(roll, abs=1e-10)
        assert p2 == pytest.approx(pitch, abs=1e-10)
        assert y2 == pytest.approx(yaw, abs=1e-10)

    def test_rotation_matrix_round_trip(self) -> None:
        """to_rotation_matrix → from_rotation_matrix should recover."""
        q = Quaternion.from_euler(0.1, 0.4, -0.3)
        R = q.to_rotation_matrix()
        q2 = Quaternion.from_rotation_matrix(R)
        # Quaternions may differ by sign (q and -q represent same rotation)
        dot = q.w * q2.w + q.x * q2.x + q.y * q2.y + q.z * q2.z
        assert abs(abs(dot) - 1.0) < 1e-10

    def test_rotate_vector_identity(self) -> None:
        q = Quaternion.identity()
        v = np.array([1.0, 2.0, 3.0])
        result = q.rotate_vector(v)
        np.testing.assert_allclose(result, v, atol=1e-10)

    def test_rotate_vector_90_z(self) -> None:
        """90° rotation about Z should map [1,0,0] → [0,1,0]."""
        q = Quaternion.from_euler(0.0, 0.0, math.pi / 2)
        v = np.array([1.0, 0.0, 0.0])
        result = q.rotate_vector(v)
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-10)

    def test_rotation_matrix_matches_vector_rotation(self) -> None:
        """R @ v should equal q.rotate_vector(v)."""
        q = Quaternion.from_euler(0.5, 0.3, 0.7)
        R = q.to_rotation_matrix()
        v = np.array([1.0, -2.0, 3.0])
        np.testing.assert_allclose(R @ v, q.rotate_vector(v), atol=1e-10)


class TestSlerp:
    """Tests for spherical linear interpolation."""

    def test_slerp_endpoints(self) -> None:
        q0 = Quaternion.identity()
        q1 = Quaternion.from_euler(0.0, 0.0, math.pi / 2)
        r0 = slerp(q0, q1, 0.0)
        r1 = slerp(q0, q1, 1.0)
        assert r0.w == pytest.approx(q0.w, abs=1e-10)
        assert r1.w == pytest.approx(q1.w, abs=1e-10)

    def test_slerp_midpoint(self) -> None:
        q0 = Quaternion.identity()
        q1 = Quaternion.from_euler(0.0, 0.0, math.pi / 2)
        mid = slerp(q0, q1, 0.5)
        _, _, yaw = mid.to_euler()
        assert yaw == pytest.approx(math.pi / 4, abs=1e-6)

    def test_slerp_invalid_t(self) -> None:
        q = Quaternion.identity()
        with pytest.raises(ValueError):
            slerp(q, q, -0.1)
        with pytest.raises(ValueError):
            slerp(q, q, 1.1)
