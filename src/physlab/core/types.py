"""Shared poses, joint state, and motion results.

Positions are meters, angles are radians, and every pose names the reference
frame it is expressed in. The default frame is ``world``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from physlab.core.errors import PhysLabError
from physlab.core.frames import matrix_to_quat, quat_angle_between, quat_normalize, quat_to_matrix

CHILD_FRAME = "body"
"""Frame tag for a pose expressed in its parent's body frame.

:meth:`Pose.compose` requires the right-hand pose to use this frame.
"""


class Mode(StrEnum):
    """Where an action runs."""

    SIM = "sim"
    REAL = "real"


class FailureReason(StrEnum):
    """Why a motion did not succeed."""

    IK_FAILED = "IK_FAILED"
    PLANNING_FAILED = "PLANNING_FAILED"
    COLLISION = "COLLISION"
    JOINT_LIMIT = "JOINT_LIMIT"
    TIMEOUT = "TIMEOUT"
    SAFETY = "SAFETY"
    HARDWARE = "HARDWARE"
    PERCEPTION = "PERCEPTION"
    GRASP_FAILED = "GRASP_FAILED"
    TOLERANCE = "TOLERANCE"


class Pose(BaseModel):
    """Rigid transform from a body frame into ``frame``.

    ``to_matrix`` returns ``T`` such that ``p_frame = T @ p_body`` for a
    homogeneous point. The stored quaternion is normalized and canonical.

    Child-frame convention for :meth:`compose`: ``result = self * other``
    means ``T_result = T_self @ T_other``. ``self`` maps its body frame into
    ``self.frame``. ``other`` must map its own body frame into ``self``'s body
    frame, which callers mark by setting ``other.frame`` to :data:`CHILD_FRAME`
    (``"body"``). The result keeps ``self.frame``.

    Attributes:
        position_m: Body origin in ``frame``, meters.
        quat_wxyz: Canonical unit quaternion ``(w, x, y, z)``.
        frame: Reference frame this pose is expressed in.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    position_m: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    frame: str = "world"

    @field_validator("position_m", mode="before")
    @classmethod
    def _position(cls, value: object) -> tuple[float, float, float]:
        vector = _as_floats(value, 3, "position_m")
        return (float(vector[0]), float(vector[1]), float(vector[2]))

    @field_validator("quat_wxyz", mode="before")
    @classmethod
    def _quaternion(cls, value: object) -> tuple[float, float, float, float]:
        try:
            normalized = quat_normalize(np.asarray(value, dtype=np.float64))
        except (TypeError, ValueError) as exc:
            raise PhysLabError(
                "quat_wxyz must be a numeric quaternion",
                details={"reason": str(exc)},
            ) from exc
        return (
            float(normalized[0]),
            float(normalized[1]),
            float(normalized[2]),
            float(normalized[3]),
        )

    @field_validator("frame")
    @classmethod
    def _frame_name(cls, value: str) -> str:
        return _require_frame(value)

    def to_matrix(self) -> npt.NDArray[np.float64]:
        """Return the homogeneous 4x4 matrix for this pose.

        Returns:
            Transform that maps body-frame points into ``frame``.
        """
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = quat_to_matrix(np.asarray(self.quat_wxyz, dtype=np.float64))
        transform[:3, 3] = np.asarray(self.position_m, dtype=np.float64)
        return transform

    @classmethod
    def from_matrix(cls, matrix: npt.ArrayLike, frame: str = "world") -> Self:
        """Build a pose from a homogeneous 4x4 matrix.

        Args:
            matrix: Transform mapping body-frame points into ``frame``.
            frame: Reference frame name stored on the result.

        Returns:
            Pose with a canonical quaternion.

        Raises:
            PhysLabError: ``matrix`` is not a finite 4x4 rigid transform, or
                ``frame`` is empty.
        """
        transform = _as_transform(matrix)
        translation = transform[:3, 3]
        quaternion = matrix_to_quat(transform[:3, :3])
        return cls(
            position_m=(float(translation[0]), float(translation[1]), float(translation[2])),
            quat_wxyz=(
                float(quaternion[0]),
                float(quaternion[1]),
                float(quaternion[2]),
                float(quaternion[3]),
            ),
            frame=frame,
        )

    def compose(self, other: Pose) -> Pose:
        """Compose ``self * other`` under the child-frame convention.

        Args:
            other: Pose of a further body, expressed in this pose's body frame.
                ``other.frame`` must equal :data:`CHILD_FRAME`.

        Returns:
            Pose of ``other``'s body expressed in ``self.frame``.

        Raises:
            PhysLabError: ``other.frame`` is not the child frame.
        """
        if other.frame != CHILD_FRAME:
            raise PhysLabError(
                "Child-frame convention violated: other.frame must be 'body'",
                details={"other_frame": other.frame, "expected_frame": CHILD_FRAME},
            )
        return Pose.from_matrix(self.to_matrix() @ other.to_matrix(), frame=self.frame)

    def inverse(self) -> Pose:
        """Return the inverse transform, tagged as a child-frame pose.

        The result maps points from ``self.frame`` back into this body. Its
        ``frame`` is :data:`CHILD_FRAME`, so ``self.compose(self.inverse())``
        is the identity in ``self.frame``.

        Returns:
            Inverse pose expressed in the body frame.
        """
        rotation = quat_to_matrix(np.asarray(self.quat_wxyz, dtype=np.float64))
        translation = np.asarray(self.position_m, dtype=np.float64)
        inverse_rotation = rotation.T
        inverse_translation = -inverse_rotation @ translation
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = inverse_rotation
        transform[:3, 3] = inverse_translation
        return Pose.from_matrix(transform, frame=CHILD_FRAME)

    def error_to(self, other: Pose) -> tuple[float, float]:
        """Measure translation and rotation from this pose to ``other``.

        Both poses must name the same reference frame. The translation error
        is the Euclidean distance between origins. The rotation error is the
        geodesic angle between orientations.

        Args:
            other: Pose to compare against, in the same frame.

        Returns:
            ``(translation_m, rotation_rad)``.

        Raises:
            PhysLabError: The poses name different frames.
        """
        if other.frame != self.frame:
            raise PhysLabError(
                "Poses must share a frame to compute error",
                details={"frame": self.frame, "other_frame": other.frame},
            )
        delta = np.asarray(other.position_m, dtype=np.float64) - np.asarray(
            self.position_m, dtype=np.float64
        )
        translation_m = float(np.linalg.norm(delta))
        rotation_rad = quat_angle_between(
            np.asarray(self.quat_wxyz, dtype=np.float64),
            np.asarray(other.quat_wxyz, dtype=np.float64),
        )
        return translation_m, rotation_rad

    def with_offset(self, dx_m: float, dy_m: float, dz_m: float) -> Pose:
        """Shift the origin by a vector expressed in this pose's body frame.

        Orientation and ``frame`` are unchanged. The offset is rotated into
        ``frame`` before it is added to ``position_m``.

        Args:
            dx_m: Body-frame X offset, meters.
            dy_m: Body-frame Y offset, meters.
            dz_m: Body-frame Z offset, meters.

        Returns:
            New pose with the shifted origin.

        Raises:
            PhysLabError: An offset is non-finite.
        """
        offset = _as_floats((dx_m, dy_m, dz_m), 3, "offset_m")
        rotation = quat_to_matrix(np.asarray(self.quat_wxyz, dtype=np.float64))
        updated = np.asarray(self.position_m, dtype=np.float64) + rotation @ offset
        return Pose(
            position_m=(float(updated[0]), float(updated[1]), float(updated[2])),
            quat_wxyz=self.quat_wxyz,
            frame=self.frame,
        )


class Pose2D(BaseModel):
    """Planar pose in a named frame.

    Attributes:
        x_m: X position, meters.
        y_m: Y position, meters.
        yaw_rad: Heading about Z, radians.
        frame: Reference frame. Defaults to ``world``.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    x_m: float
    y_m: float
    yaw_rad: float
    frame: str = "world"

    @field_validator("x_m", "y_m", "yaw_rad")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not np.isfinite(value):
            raise PhysLabError("Pose2D values must be finite")
        return value

    @field_validator("frame")
    @classmethod
    def _frame_name(cls, value: str) -> str:
        return _require_frame(value)


class JointState(BaseModel):
    """Named joint positions and optional velocities.

    Attributes:
        names: Joint names, in the same order as the position vector.
        positions_rad: Joint positions, radians.
        velocities_rad_s: Joint velocities, radians per second, or ``None``.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    names: tuple[str, ...]
    positions_rad: tuple[float, ...]
    velocities_rad_s: tuple[float, ...] | None = None

    @model_validator(mode="after")
    def _check_lengths(self) -> Self:
        count = len(self.names)
        if len(self.positions_rad) != count:
            raise PhysLabError(
                "positions_rad length must match names",
                details={"names": count, "positions_rad": len(self.positions_rad)},
            )
        if self.velocities_rad_s is not None and len(self.velocities_rad_s) != count:
            raise PhysLabError(
                "velocities_rad_s length must match names",
                details={"names": count, "velocities_rad_s": len(self.velocities_rad_s)},
            )
        if not all(np.isfinite(position) for position in self.positions_rad):
            raise PhysLabError("positions_rad must be finite")
        if self.velocities_rad_s is not None and not all(
            np.isfinite(velocity) for velocity in self.velocities_rad_s
        ):
            raise PhysLabError("velocities_rad_s must be finite")
        return self


class MotionResult(BaseModel):
    """Outcome of one motion command.

    Attributes:
        success: Whether the motion reached its target.
        reason: Failure category. ``None`` when the caller has no category.
        message: Extra detail. Empty when unused.
        final_pos_error_m: Translation error at the end of the motion, meters.
        final_rot_error_rad: Rotation error at the end of the motion, radians.
        duration_s: Time spent executing the motion, seconds.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    success: bool
    reason: FailureReason | None
    message: str = ""
    final_pos_error_m: float
    final_rot_error_rad: float
    duration_s: float

    @model_validator(mode="after")
    def _check_metrics(self) -> Self:
        metrics = (self.final_pos_error_m, self.final_rot_error_rad, self.duration_s)
        if not all(np.isfinite(metric) for metric in metrics):
            raise PhysLabError("Motion errors and duration must be finite")
        if any(metric < 0.0 for metric in metrics):
            raise PhysLabError("Motion errors and duration must be non-negative")
        return self


def _require_frame(value: str) -> str:
    if not value:
        raise PhysLabError("Frame name must be non-empty")
    return value


def _as_floats(value: object, length: int, name: str) -> npt.NDArray[np.float64]:
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


def _as_transform(matrix: npt.ArrayLike) -> npt.NDArray[np.float64]:
    try:
        array = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PhysLabError(
            "Transform must be a numeric matrix",
            details={"reason": str(exc)},
        ) from exc
    if array.shape != (4, 4):
        raise PhysLabError(
            "Transform must have shape (4, 4)",
            details={"shape": [int(dim) for dim in array.shape]},
        )
    if not bool(np.all(np.isfinite(array))):
        raise PhysLabError("Transform must be finite")
    return array
