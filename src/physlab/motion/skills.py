"""Robot-agnostic motions built only from the HAL.

``safe_transit`` lifts to a clearance height, moves across at that height, and
descends. ``approach`` and ``retreat`` travel along the tool Z axis.
``navigate_and_reach`` drives a mobile base and then reaches with the arm.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from physlab.core.errors import ConfigError
from physlab.core.frames import quat_wxyz_to_xyzw, quat_xyzw_to_wxyz
from physlab.core.types import MotionResult, Pose, Pose2D
from physlab.hal.interfaces import Arm, Robot

_STEP_M = 0.02
# Joint tracking undershoots a Cartesian waypoint by about a millimeter.
# Command the transit slightly above the requested clearance so the executed
# path stays at or above that clearance.
_CLEARANCE_MARGIN_M = 0.002


def safe_transit(robot: Robot, target: Pose, clearance_z_m: float) -> MotionResult:
    """Lift, travel at ``clearance_z_m``, then descend to ``target``.

    Args:
        robot: Robot whose arm receives the pose goals.
        target: Final end-effector pose.
        clearance_z_m: World-frame height of the transit, meters.

    Returns:
        Combined motion result. A failed segment is returned as-is.
    """
    arm = _require_arm(robot)
    current = arm.get_ee_pose()
    height = clearance_z_m + _CLEARANCE_MARGIN_M
    lift = _at(current, current.position_m[0], current.position_m[1], height, current)
    hover = _at(target, target.position_m[0], target.position_m[1], height, target)
    return _follow(arm, [current, lift, hover, target])


def approach(robot: Robot, target: Pose, standoff_m: float) -> MotionResult:
    """Move to a standoff along the tool -Z axis, then to ``target``.

    The tool +Z axis is the target pose's body Z. The standoff sits
    ``standoff_m`` behind the target, and the second motion drives forward.

    Args:
        robot: Robot whose arm moves.
        target: Pose at the end of the approach.
        standoff_m: Distance behind the target along -Z, meters.

    Returns:
        Combined motion result.
    """
    arm = _require_arm(robot)
    standoff = target.with_offset(0.0, 0.0, -standoff_m)
    return _chain(
        [arm.move_pose(standoff, speed_scale=1.0), arm.move_pose(target, speed_scale=1.0)]
    )


def retreat(robot: Robot, distance_m: float) -> MotionResult:
    """Back away from the current pose along the tool -Z axis.

    Args:
        robot: Robot whose arm moves.
        distance_m: Distance to travel, meters.

    Returns:
        Motion result of the retreat.
    """
    arm = _require_arm(robot)
    goal = arm.get_ee_pose().with_offset(0.0, 0.0, -distance_m)
    return arm.move_pose(goal, speed_scale=1.0)


def navigate_and_reach(robot: Robot, base_goal: Pose2D, ee_target: Pose) -> MotionResult:
    """Drive the base to ``base_goal``, then move the arm to ``ee_target``.

    Args:
        robot: Mobile manipulator.
        base_goal: Planar goal for the base.
        ee_target: End-effector goal in the world frame after the drive.

    Returns:
        Combined motion result.

    Raises:
        ConfigError: The robot has no base or no arm.
    """
    if robot.base is None or robot.arm is None:
        raise ConfigError("navigate_and_reach needs a base and an arm")
    driven = robot.base.move_to(base_goal)
    if not driven.success:
        return driven
    reached = robot.arm.move_pose(ee_target, speed_scale=1.0, timeout_s=20.0)
    return _chain([driven, reached])


def _require_arm(robot: Robot) -> Arm:
    if robot.arm is None:
        raise ConfigError("Skill needs an arm")
    return robot.arm


def _follow(arm: Arm, poses: list[Pose]) -> MotionResult:
    results: list[MotionResult] = []
    for start, end in itertools.pairwise(poses):
        for pose in _samples(start, end):
            result = arm.move_pose(pose, speed_scale=1.0, timeout_s=20.0)
            results.append(result)
            if not result.success:
                return _chain(results)
    return _chain(results)


def _samples(start: Pose, end: Pose) -> list[Pose]:
    span = math.dist(start.position_m, end.position_m)
    count = max(1, math.ceil(span / _STEP_M))
    return [_blend(start, end, (index + 1) / count) for index in range(count)]


def _at(pose: Pose, x_m: float, y_m: float, z_m: float, orientation: Pose) -> Pose:
    return Pose(position_m=(x_m, y_m, z_m), quat_wxyz=orientation.quat_wxyz, frame=pose.frame)


def _blend(start: Pose, end: Pose, fraction: float) -> Pose:
    position = (1.0 - fraction) * np.asarray(start.position_m) + fraction * np.asarray(
        end.position_m
    )
    ends = np.vstack(
        [
            quat_wxyz_to_xyzw(np.asarray(start.quat_wxyz)),
            quat_wxyz_to_xyzw(np.asarray(end.quat_wxyz)),
        ]
    )
    quat = quat_xyzw_to_wxyz(Slerp([0.0, 1.0], Rotation.from_quat(ends))([fraction]).as_quat()[0])
    return Pose(
        position_m=(float(position[0]), float(position[1]), float(position[2])),
        quat_wxyz=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
        frame=start.frame,
    )


def _chain(results: list[MotionResult]) -> MotionResult:
    if not results:
        raise ConfigError("Skill produced no motion")
    failed = next((item for item in results if not item.success), None)
    chosen = failed or results[-1]
    duration = sum(item.duration_s for item in results)
    return MotionResult(
        success=chosen.success,
        reason=chosen.reason,
        message=chosen.message,
        final_pos_error_m=chosen.final_pos_error_m,
        final_rot_error_rad=chosen.final_rot_error_rad,
        duration_s=duration,
    )
