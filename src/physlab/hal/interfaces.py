"""Synchronous hardware interfaces shared by simulation and real robots.

``move_pose`` is part of the arm contract. The simulation backend raises
``NotImplementedError`` until inverse kinematics lands.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from physlab.core.types import JointState, MotionResult, Pose, Pose2D
from physlab.robots.models import RobotSpec


class CameraIntrinsics(BaseModel):
    """Pinhole intrinsics in pixels.

    Attributes:
        fx: Focal length along x, pixels.
        fy: Focal length along y, pixels.
        cx: Principal point x, pixels.
        cy: Principal point y, pixels.
        width: Image width, pixels.
        height: Image height, pixels.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    fx: float
    fy: float
    cx: float
    cy: float
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Frame(BaseModel):
    """One camera capture.

    Attributes:
        rgb: Unsigned 8-bit image, shape ``(height, width, 3)``.
        depth_m: Depth in meters, shape ``(height, width)``, or ``None``.
        intrinsics: Pinhole parameters used for this image.
        pose_in_world: Camera pose in the world frame, if known.
        timestamp_s: Simulation or sensor time, seconds.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid", arbitrary_types_allowed=True
    )

    rgb: npt.NDArray[np.uint8]
    depth_m: npt.NDArray[np.float32] | None
    intrinsics: CameraIntrinsics
    pose_in_world: Pose | None
    timestamp_s: float


class Contact(BaseModel):
    """One reported contact between two geoms.

    Attributes:
        geom1: First geom name. Empty when the geom is unnamed.
        geom2: Second geom name.
        body1: Body that owns the first geom.
        body2: Body that owns the second geom.
        dist: Signed separation, meters. Negative values penetrate.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    geom1: str
    geom2: str
    body1: str
    body2: str
    dist: float


@runtime_checkable
class Arm(Protocol):
    """Joint-space arm. Pose goals are filled in by a later task."""

    name: str
    dof: int
    joint_names: tuple[str, ...]

    def get_joint_state(self) -> JointState:
        """Return positions and velocities for ``joint_names``."""

    def get_ee_pose(self) -> Pose:
        """Return the end-effector pose in the world frame."""

    def move_joints(
        self,
        q_rad: Sequence[float],
        *,
        speed_scale: float = 0.5,
        timeout_s: float = 10.0,
    ) -> MotionResult:
        """Move joints toward ``q_rad`` and return when settled or blocked."""

    def move_pose(
        self,
        target: Pose,
        *,
        speed_scale: float = 0.5,
        timeout_s: float = 10.0,
    ) -> MotionResult:
        """Move the end effector to ``target``.

        Raises:
            NotImplementedError: Always, until inverse kinematics exists.
        """

    def stop(self) -> None:
        """Hold the current joint position."""


@runtime_checkable
class Gripper(Protocol):
    """Parallel or angular gripper commanded in its own control units."""

    def open(self) -> MotionResult:
        """Open the fingers."""

    def close(self, force_scale: float = 0.5) -> MotionResult:
        """Close the fingers. ``force_scale`` is reserved for later force limits."""

    def get_opening(self) -> float:
        """Return the opening fraction, 0 closed and 1 open."""

    def is_grasping(self) -> bool:
        """Return whether both finger groups touch the same external body."""


@runtime_checkable
class MobileBase(Protocol):
    """Planar mobile base."""

    def get_pose2d(self) -> Pose2D:
        """Return x, y, and yaw in the world frame."""

    def set_velocity(
        self,
        vx_m_s: float,
        vy_m_s: float,
        wz_rad_s: float,
        duration_s: float,
    ) -> MotionResult:
        """Hold a body velocity for ``duration_s`` seconds."""

    def move_to(
        self,
        target: Pose2D,
        *,
        pos_tol_m: float = 0.02,
        yaw_tol_rad: float = 0.05,
        timeout_s: float = 30.0,
    ) -> MotionResult:
        """Drive to ``target`` within the given tolerances."""

    def stop(self) -> None:
        """Zero the base command."""


@runtime_checkable
class Camera(Protocol):
    """RGB and optional depth camera."""

    name: str

    def capture(self) -> Frame:
        """Grab one frame from the current simulation state."""


@dataclass
class Robot:
    """One placed robot and the devices attached to it.

    Attributes:
        spec: Catalog entry for this robot id.
        arm: Arm interface, or ``None`` when the robot has no arm joints.
        gripper: Gripper interface, or ``None``.
        base: Mobile base interface, or ``None``.
        cameras: Cameras keyed by name.
    """

    spec: RobotSpec
    arm: Arm | None
    gripper: Gripper | None
    base: MobileBase | None
    cameras: dict[str, Camera]
