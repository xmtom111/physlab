"""Quaternion and rotation helpers.

Quaternions in physlab are always ``(w, x, y, z)``, matching MuJoCo.
``scipy.spatial.transform.Rotation`` stores ``(x, y, z, w)``. Convert only with
:func:`quat_wxyz_to_xyzw` and :func:`quat_xyzw_to_wxyz`.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from physlab.core.errors import PhysLabError

ArrayLike = npt.ArrayLike


def quat_wxyz_to_xyzw(q: ArrayLike) -> npt.NDArray[np.float64]:
    """Reorder a quaternion from ``(w, x, y, z)`` to ``(x, y, z, w)``.

    The values are copied and are not normalized.

    Args:
        q: Quaternion in ``(w, x, y, z)`` order.

    Returns:
        Quaternion in ``(x, y, z, w)`` order.

    Raises:
        PhysLabError: ``q`` does not have shape ``(4,)`` or is non-finite.
    """
    vector = _as_quat(q)
    return np.array([vector[1], vector[2], vector[3], vector[0]], dtype=np.float64)


def quat_xyzw_to_wxyz(q: ArrayLike) -> npt.NDArray[np.float64]:
    """Reorder a quaternion from ``(x, y, z, w)`` to ``(w, x, y, z)``.

    The values are copied and are not normalized.

    Args:
        q: Quaternion in ``(x, y, z, w)`` order.

    Returns:
        Quaternion in ``(w, x, y, z)`` order.

    Raises:
        PhysLabError: ``q`` does not have shape ``(4,)`` or is non-finite.
    """
    vector = _as_quat(q)
    return np.array([vector[3], vector[0], vector[1], vector[2]], dtype=np.float64)


def quat_normalize(q: ArrayLike) -> npt.NDArray[np.float64]:
    """Return a unit quaternion with a canonical sign.

    The canonical representative has its first non-zero component positive.
    When ``w`` is not zero this means ``w >= 0``, so ``q`` and ``-q`` normalize
    to the same rotation.

    Args:
        q: Quaternion in ``(w, x, y, z)`` order.

    Returns:
        Unit quaternion in ``(w, x, y, z)`` order.

    Raises:
        PhysLabError: ``q`` does not have shape ``(4,)``, is non-finite, or has
            zero norm.
    """
    vector = _as_quat(q)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise PhysLabError("Quaternion has zero norm", details={"norm": norm})
    unit = vector / norm
    return _canonicalize(unit)


def quat_multiply(a: ArrayLike, b: ArrayLike) -> npt.NDArray[np.float64]:
    """Hamilton product ``a * b`` in ``(w, x, y, z)`` order.

    The product matches matrix composition: ``R(a * b) = R(a) @ R(b)``.
    Both inputs are normalized, and the result is canonical.

    Args:
        a: Left quaternion, ``(w, x, y, z)``.
        b: Right quaternion, ``(w, x, y, z)``.

    Returns:
        Canonical unit quaternion ``a * b``.

    Raises:
        PhysLabError: Either input is not a finite quaternion of shape ``(4,)``
            or has zero norm.
    """
    left = quat_normalize(a)
    right = quat_normalize(b)
    product = _hamilton(left, right)
    return quat_normalize(product)


def quat_to_matrix(q: ArrayLike) -> npt.NDArray[np.float64]:
    """Convert a ``(w, x, y, z)`` quaternion to a 3x3 rotation matrix.

    Args:
        q: Quaternion in ``(w, x, y, z)`` order. It is normalized first.

    Returns:
        Rotation matrix ``R`` such that ``v_parent = R @ v_body``.

    Raises:
        PhysLabError: ``q`` is not a finite non-zero quaternion of shape ``(4,)``.
    """
    xyzw = quat_wxyz_to_xyzw(quat_normalize(q))
    matrix = Rotation.from_quat(xyzw).as_matrix()
    return np.asarray(matrix, dtype=np.float64)


def matrix_to_quat(rotation: ArrayLike) -> npt.NDArray[np.float64]:
    """Convert a 3x3 rotation matrix to a canonical ``(w, x, y, z)`` quaternion.

    Args:
        rotation: Rotation matrix with shape ``(3, 3)``.

    Returns:
        Canonical unit quaternion in ``(w, x, y, z)`` order.

    Raises:
        PhysLabError: ``rotation`` is not a finite 3x3 rotation matrix.
    """
    matrix = _as_square(rotation, 3, "rotation")
    try:
        xyzw = Rotation.from_matrix(matrix).as_quat()
    except ValueError as exc:
        raise PhysLabError(
            "Rotation matrix is invalid",
            details={"reason": str(exc)},
        ) from exc
    return quat_normalize(quat_xyzw_to_wxyz(xyzw))


def rpy_to_quat(roll_rad: float, pitch_rad: float, yaw_rad: float) -> npt.NDArray[np.float64]:
    """Convert intrinsic XYZ roll-pitch-yaw to a quaternion.

    The rotation is applied in the moving frame: roll about X, then pitch
    about the new Y, then yaw about the new Z. A yaw-only angle is therefore
    a rotation about the fixed Z axis.

    Args:
        roll_rad: Rotation about X, radians.
        pitch_rad: Rotation about Y, radians.
        yaw_rad: Rotation about Z, radians.

    Returns:
        Canonical unit quaternion in ``(w, x, y, z)`` order.

    Raises:
        PhysLabError: An angle is non-finite.
    """
    angles = (roll_rad, pitch_rad, yaw_rad)
    if not all(np.isfinite(angle) for angle in angles):
        raise PhysLabError("Roll, pitch, and yaw must be finite", details={"rpy_rad": angles})
    xyzw = Rotation.from_euler("xyz", [roll_rad, pitch_rad, yaw_rad]).as_quat()
    return quat_normalize(quat_xyzw_to_wxyz(xyzw))


def quat_angle_between(a: ArrayLike, b: ArrayLike) -> float:
    """Return the geodesic rotation angle between two quaternions, in radians.

    The angle is in ``[0, pi]``. ``q`` and ``-q`` are the same rotation.
    The value is ``2 * atan2(||v||, |w|)`` of the relative quaternion, which
    stays accurate for tiny angles.

    Args:
        a: First quaternion, ``(w, x, y, z)``.
        b: Second quaternion, ``(w, x, y, z)``.

    Returns:
        Rotation angle in radians.

    Raises:
        PhysLabError: Either input is not a finite non-zero quaternion.
    """
    left = quat_normalize(a)
    right = quat_normalize(b)
    conjugate = left.copy()
    conjugate[1:] *= -1.0
    relative = _hamilton(conjugate, right)
    # abs(w) selects the short arc, so q and -q stay the same rotation.
    half_cos = abs(float(relative[0]))
    half_sin = float(np.linalg.norm(relative[1:]))
    return float(2.0 * math.atan2(half_sin, half_cos))


def _as_quat(q: ArrayLike) -> npt.NDArray[np.float64]:
    return _as_vector(q, 4, "quaternion")


def _as_vector(value: ArrayLike, length: int, name: str) -> npt.NDArray[np.float64]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PhysLabError(
            f"{name} must be a numeric vector",
            details={"reason": str(exc)},
        ) from exc
    if array.shape != (length,):
        raise PhysLabError(
            f"{name} must have shape ({length},)",
            details={"shape": [int(dim) for dim in array.shape]},
        )
    if not bool(np.all(np.isfinite(array))):
        raise PhysLabError(f"{name} must be finite")
    return array


def _as_square(value: ArrayLike, size: int, name: str) -> npt.NDArray[np.float64]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PhysLabError(
            f"{name} must be a numeric matrix",
            details={"reason": str(exc)},
        ) from exc
    if array.shape != (size, size):
        raise PhysLabError(
            f"{name} must have shape ({size}, {size})",
            details={"shape": [int(dim) for dim in array.shape]},
        )
    if not bool(np.all(np.isfinite(array))):
        raise PhysLabError(f"{name} must be finite")
    return array


def _canonicalize(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    for component in q:
        if float(component) > 1e-12:
            return q
        if float(component) < -1e-12:
            return -q
    return q


def _hamilton(
    left: npt.NDArray[np.float64],
    right: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    lw, lx, ly, lz = (float(component) for component in left)
    rw, rx, ry, rz = (float(component) for component in right)
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )
