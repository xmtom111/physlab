"""Pose algebra, joint state, and motion results."""

from __future__ import annotations

import numpy as np
import pytest

from physlab.core.errors import PhysLabError
from physlab.core.frames import quat_normalize, rpy_to_quat
from physlab.core.types import (
    CHILD_FRAME,
    FailureReason,
    JointState,
    Mode,
    MotionResult,
    Pose,
    Pose2D,
)


def test_compose_inverse_and_matrix_round_trip() -> None:
    rng = np.random.default_rng(1)
    origin = Pose(position_m=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0), frame="world")
    for _ in range(50):
        position = tuple(float(value) for value in rng.uniform(-3.0, 3.0, size=3))
        quaternion = tuple(float(value) for value in quat_normalize(rng.normal(size=4)))
        pose = Pose(position_m=position, quat_wxyz=quaternion, frame="world")
        identity = pose.compose(pose.inverse())
        translation_m, rotation_rad = identity.error_to(origin)
        assert translation_m < 1e-9
        assert rotation_rad < 1e-9
        assert identity.frame == "world"
        assert pose.inverse().frame == CHILD_FRAME
        restored = Pose.from_matrix(pose.to_matrix(), frame="world")
        translation_m, rotation_rad = pose.error_to(restored)
        assert translation_m < 1e-9
        assert rotation_rad < 1e-9


def test_compose_translation_and_body_offset() -> None:
    base = Pose(position_m=(1.0, 2.0, 3.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0))
    child = Pose(position_m=(0.5, 0.0, -0.25), quat_wxyz=(1.0, 0.0, 0.0, 0.0), frame=CHILD_FRAME)
    composed = base.compose(child)
    assert composed.position_m == pytest.approx((1.5, 2.0, 2.75))
    assert composed.frame == "world"
    yaw = Pose(
        position_m=(0.0, 0.0, 0.0),
        quat_wxyz=tuple(float(value) for value in rpy_to_quat(0.0, 0.0, np.pi / 2)),
    )
    shifted = yaw.with_offset(1.0, 0.0, 0.0)
    assert shifted.position_m == pytest.approx((0.0, 1.0, 0.0))
    assert shifted.quat_wxyz == yaw.quat_wxyz


def test_pose_rejects_bad_values() -> None:
    with pytest.raises(PhysLabError):
        Pose(position_m=(0.0, 0.0, 0.0), quat_wxyz=(0.0, 0.0, 0.0, 0.0))
    with pytest.raises(PhysLabError):
        Pose(position_m=(0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0))
    with pytest.raises(PhysLabError):
        Pose(position_m=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0), frame="")
    flipped = Pose(position_m=(0.0, 0.0, 0.0), quat_wxyz=(-1.0, 0.0, 0.0, 0.0))
    assert flipped.quat_wxyz == (1.0, 0.0, 0.0, 0.0)
    world = Pose(position_m=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0), frame="world")
    body = Pose(position_m=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0), frame=CHILD_FRAME)
    with pytest.raises(PhysLabError):
        world.compose(world)
    with pytest.raises(PhysLabError):
        world.error_to(body)
    with pytest.raises(PhysLabError):
        Pose.from_matrix(np.eye(3))
    with pytest.raises(PhysLabError):
        world.with_offset(np.nan, 0.0, 0.0)


def test_pose2d_joint_state_and_motion_result() -> None:
    planar = Pose2D(x_m=1.0, y_m=-2.0, yaw_rad=0.3)
    assert planar.frame == "world"
    with pytest.raises(PhysLabError):
        Pose2D(x_m=np.nan, y_m=0.0, yaw_rad=0.0)
    state = JointState(names=("j1", "j2"), positions_rad=(0.1, -0.2), velocities_rad_s=(0.0, 0.5))
    assert state.names == ("j1", "j2")
    with pytest.raises(PhysLabError):
        JointState(names=("j1",), positions_rad=(0.1, 0.2))
    with pytest.raises(PhysLabError):
        JointState(names=("j1",), positions_rad=(0.1,), velocities_rad_s=())
    ok = MotionResult(
        success=False,
        reason=FailureReason.IK_FAILED,
        final_pos_error_m=0.01,
        final_rot_error_rad=0.02,
        duration_s=1.5,
    )
    assert ok.reason is FailureReason.IK_FAILED
    assert Mode.SIM.value == "sim"
    with pytest.raises(PhysLabError):
        MotionResult(
            success=True,
            reason=None,
            final_pos_error_m=-1.0,
            final_rot_error_rad=0.0,
            duration_s=0.0,
        )
