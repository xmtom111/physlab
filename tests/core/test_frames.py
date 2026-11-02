"""Quaternion conversion properties."""

from __future__ import annotations

import numpy as np
import pytest

from physlab.core.errors import PhysLabError
from physlab.core.frames import (
    matrix_to_quat,
    quat_angle_between,
    quat_multiply,
    quat_normalize,
    quat_to_matrix,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
    rpy_to_quat,
)


def test_reorder_is_exact() -> None:
    raw = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.array_equal(quat_wxyz_to_xyzw(raw), np.array([2.0, 3.0, 4.0, 1.0]))
    assert np.array_equal(quat_xyzw_to_wxyz(np.array([2.0, 3.0, 4.0, 1.0])), raw)


def test_normalize_canonical_sign_and_unit_length() -> None:
    normalized = quat_normalize(np.array([-2.0, 0.0, 0.0, 0.0]))
    assert np.allclose(normalized, np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.isclose(np.linalg.norm(quat_normalize([1.0, 2.0, 3.0, 4.0])), 1.0)


def test_quaternion_rejects_bad_input() -> None:
    with pytest.raises(PhysLabError):
        quat_normalize([0.0, 0.0, 0.0, 0.0])
    with pytest.raises(PhysLabError):
        quat_wxyz_to_xyzw([1.0, 0.0, 0.0])
    with pytest.raises(PhysLabError):
        quat_normalize([np.nan, 0.0, 0.0, 0.0])
    with pytest.raises(PhysLabError):
        matrix_to_quat(np.zeros((3, 3)))
    with pytest.raises(PhysLabError):
        matrix_to_quat(np.eye(4))
    with pytest.raises(PhysLabError):
        rpy_to_quat(0.0, np.inf, 0.0)


def test_known_yaw_and_angle() -> None:
    yaw = quat_to_matrix(rpy_to_quat(0.0, 0.0, np.pi / 2))
    expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(yaw, expected, atol=1e-9)
    assert quat_angle_between([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert quat_angle_between([1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert quat_angle_between([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]) == pytest.approx(np.pi)
    turned = quat_multiply(rpy_to_quat(0.0, 0.0, np.pi / 2), rpy_to_quat(0.0, 0.0, np.pi / 2))
    assert quat_angle_between(turned, rpy_to_quat(0.0, 0.0, np.pi)) < 1e-9


def test_random_quaternion_round_trips() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        raw = _nonzero(rng)
        assert np.allclose(quat_xyzw_to_wxyz(quat_wxyz_to_xyzw(raw)), raw)
        canonical = quat_normalize(raw)
        recovered = matrix_to_quat(quat_to_matrix(canonical))
        assert quat_angle_between(canonical, recovered) < 1e-9
        other = _nonzero(rng)
        product = quat_multiply(canonical, other)
        composed = quat_to_matrix(canonical) @ quat_to_matrix(other)
        assert np.allclose(quat_to_matrix(product), composed, atol=1e-9)
        roll, pitch, yaw = (float(value) for value in rng.uniform(-np.pi, np.pi, size=3))
        rpy = rpy_to_quat(roll, pitch, yaw)
        assert quat_angle_between(rpy, matrix_to_quat(quat_to_matrix(rpy))) < 1e-9


def _nonzero(rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(size=4)
    if float(np.linalg.norm(raw)) < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return raw
