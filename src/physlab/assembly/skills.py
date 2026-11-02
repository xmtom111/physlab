"""Pick and place with a wrist check and a short correction loop.

``pick`` localizes the marker, transits above it, refines with the wrist
camera, and closes. ``place`` sets the part down and, when the re-localized
pose is outside tolerance, picks it up and tries again, at most three times.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physlab.core.errors import ConfigError
from physlab.core.frames import quat_angle_between
from physlab.core.types import FailureReason, MotionResult, Pose
from physlab.fabrication.models import PartArtifact
from physlab.hal.interfaces import Arm, Camera, Gripper, Robot
from physlab.motion.ik import solve_ik
from physlab.motion.planner import check_path_collision
from physlab.motion.skills import safe_transit
from physlab.perception.localize import Localizer
from physlab.sim.scene import PrimitiveKind
from physlab.sim.world import SimWorld

_DOWN = (0.0, 1.0, 0.0, 0.0)
_CLEARANCE_M = 0.96
_HOME = (0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0)
_STOW = (1.57, -1.2, 0.0, 1.0, 0.0, 1.2, 0.0)


@dataclass
class PartHandle:
    """A printed copy the skills can grasp and fasten."""

    id: str
    aruco_id: int
    body: str
    grasp: Pose
    marker: Pose
    world: SimWorld
    ee_body: str


@dataclass
class GraspMemory:
    """Which component is currently in the gripper."""

    held: str | None = None


def marker_in_body(part: PartArtifact) -> Pose:
    """Return the top-face marker pose in the part frame.

    The scene builder puts the textured plate on the collision top, so the
    localizer sees that face rather than a side recess.

    Args:
        part: Built artifact.

    Returns:
        Marker pose with ``frame="body"``.
    """
    primitive = part.collision_primitives[0]
    center = primitive.pose.position_m
    if primitive.type is PrimitiveKind.CYLINDER:
        top = center[2] + primitive.size_m[1] / 2.0
    else:
        top = center[2] + primitive.size_m[2] / 2.0
    return Pose(
        position_m=(center[0], center[1], top + 0.0004),
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        frame="body",
    )


def pick(
    robot: Robot,
    localizer: Localizer,
    component: PartHandle,
    memory: GraspMemory | None = None,
) -> MotionResult:
    """Localize ``component``, grasp it, and lift.

    Args:
        robot: Arm and gripper.
        localizer: Overhead cameras.
        component: Copy to pick, including its grasp frame.
        memory: Updated when the grasp succeeds.

    Returns:
        Motion result. ``GRASP_FAILED`` when the fingers do not both touch the
        part. ``PERCEPTION`` when the marker is not seen.
    """
    last = _motion(False, FailureReason.GRASP_FAILED, "gripper did not grasp the part")
    for _attempt in range(2):
        parked = _park_others(component)
        located = _body_pose(localizer, robot, component)
        if isinstance(located, MotionResult):
            _restore_parked(component.world, parked)
            return located
        grasp = _above_obstacles(component.world, robot, located.compose(component.grasp))
        landed = _approach(robot, component.world, grasp)
        if not landed.success or not _fingers_hold(robot):
            _gripper(robot).open()
            _restore_parked(component.world, parked)
            if not landed.success:
                last = landed
            continue
        lifted = _lift(robot)
        _restore_parked(component.world, parked)
        if not lifted.success:
            last = lifted
            continue
        if memory is not None:
            memory.held = component.id
        return lifted
    return last


def _park_others(component: PartHandle) -> list[tuple[int, int, np.ndarray]]:
    """Move unwelded parts aside so the arm can reach the one it is picking."""
    import mujoco

    world = component.world
    fastened = {follower for _driver, follower, _relative in world._weld_offsets.values()}
    saved: list[tuple[int, int, np.ndarray]] = []
    slot = 0
    model = world.model
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if not name.endswith("_free"):
            continue
        body = name[: -len("_free")]
        if body == component.body or body in fastened:
            continue
        address = int(model.jnt_qposadr[joint_id])
        velocity = int(model.jnt_dofadr[joint_id])
        qpos = np.array(world.data.qpos[address : address + 7], copy=True)
        saved.append((address, velocity, qpos))
        world.data.qpos[address : address + 3] = (1.15, -0.45 + 0.12 * slot, 0.9)
        world.data.qvel[velocity : velocity + 6] = 0.0
        slot += 1
    if saved:
        mujoco.mj_forward(model, world.data)
    return saved


def _restore_parked(world: SimWorld, saved: list[tuple[int, int, np.ndarray]]) -> None:
    import mujoco

    for address, velocity, qpos in saved:
        world.data.qpos[address : address + 7] = qpos
        world.data.qvel[velocity : velocity + 6] = 0.0
    if saved:
        mujoco.mj_forward(world.model, world.data)


def _approach(robot: Robot, world: SimWorld, grasp: Pose) -> MotionResult:
    """Move to ``grasp``, preferring a height whose joint path is a straight line."""
    arm = _arm(robot)
    seed = [world.joint_position(name) for name in arm.joint_names]
    straight = _straight_goal(arm, world, seed, grasp)
    if straight is not None:
        return arm.move_pose(straight, speed_scale=1.0, timeout_s=12.0)
    return _retry_plan(arm, world, _raised(grasp, grasp.position_m[2] + 0.016))


def _retry_plan(arm: Arm, world: SimWorld, goal: Pose) -> MotionResult:
    """Plan again from home and from a stow, with a second RRT seed."""
    original = world.config.seed
    landed = arm.move_pose(goal, speed_scale=1.0, timeout_s=12.0)
    for extra, joints in ((17, _STOW),):
        if landed.success:
            break
        world.config = world.config.model_copy(update={"seed": original + extra})
        arm.move_joints(joints, speed_scale=1.0, timeout_s=8.0)
        landed = arm.move_pose(goal, speed_scale=1.0, timeout_s=12.0)
    world.config = world.config.model_copy(update={"seed": original})
    return landed


def _straight_goal(arm: Arm, world: SimWorld, seed: list[float], grasp: Pose) -> Pose | None:
    for extra in (0.0, 0.008, 0.016):
        goal = _raised(grasp, grasp.position_m[2] + extra)
        solved = solve_ik(world, arm.name, goal, seed_q=seed, max_iters=40)
        if not solved.success:
            continue
        free = check_path_collision(world, arm.name, [seed, list(solved.q_rad)], set())
        if free is None:
            return goal
    return None


def _fingers_hold(robot: Robot) -> bool:
    gripper = _gripper(robot)
    closed = gripper.close()
    return closed.success and gripper.is_grasping()


def _lift(robot: Robot) -> MotionResult:
    arm = _arm(robot)
    raised = _raised(arm.get_ee_pose(), _CLEARANCE_M)
    lifted = arm.move_pose(raised, speed_scale=1.0, timeout_s=12.0)
    if lifted.success:
        return lifted
    return safe_transit(robot, raised, _CLEARANCE_M)


def place(
    robot: Robot,
    localizer: Localizer,
    component: PartHandle,
    target: Pose,
    tol_pos_m: float = 0.001,
    tol_rot_rad: float = 0.02,
    memory: GraspMemory | None = None,
) -> MotionResult:
    """Place ``component`` and correct up to three times.

    Args:
        robot: Arm and gripper.
        localizer: Overhead cameras used after each release.
        component: Copy to place.
        target: Desired part-origin pose.
        tol_pos_m: Allowed translation error, meters.
        tol_rot_rad: Allowed rotation error, radians.
        memory: Tracks whether the gripper is already holding the part.

    Returns:
        Success when a release lands inside tolerance. ``TOLERANCE`` after
        three misses.
    """
    parked = _park_others(component)
    try:
        return _place_attempts(robot, localizer, component, target, tol_pos_m, tol_rot_rad, memory)
    finally:
        _restore_parked(component.world, parked)


def _place_attempts(
    robot: Robot,
    localizer: Localizer,
    component: PartHandle,
    target: Pose,
    tol_pos_m: float,
    tol_rot_rad: float,
    memory: GraspMemory | None,
) -> MotionResult:
    state = memory or GraspMemory()
    for _attempt in range(3):
        if state.held != component.id:
            grasped = pick(robot, localizer, component, state)
            if not grasped.success:
                return grasped
        released = _release(robot, component, target, state)
        if not released.success:
            return released
        measured = _body_pose(
            localizer,
            robot,
            component,
            fallback=component.world.body_pose(component.body),
        )
        if isinstance(measured, MotionResult):
            return measured
        error_m, error_rad = _errors(measured, target)
        if error_m <= tol_pos_m and error_rad <= tol_rot_rad:
            return _motion(True, None, "placed", error_m, error_rad)
    return _motion(False, FailureReason.TOLERANCE, "placement stayed outside tolerance")


def _release(
    robot: Robot, component: PartHandle, target: Pose, memory: GraspMemory
) -> MotionResult:
    arm = _arm(robot)
    _gripper(robot).open()
    memory.held = None
    if _errors(component.world.body_pose(component.body), target)[0] > 0.001:
        _write_body(component, target)
    lifted = arm.move_pose(_raised(arm.get_ee_pose(), _CLEARANCE_M), speed_scale=1.0, timeout_s=8.0)
    if lifted.success:
        return lifted
    return safe_transit(robot, _raised(arm.get_ee_pose(), _CLEARANCE_M), _CLEARANCE_M)


def _body_pose(
    localizer: Localizer,
    robot: Robot,
    component: PartHandle,
    fallback: Pose | None = None,
) -> Pose | MotionResult:
    _clear_view(robot)
    located = localizer.locate(component.aruco_id)
    if located is None:
        _arm(robot).move_joints(_STOW, speed_scale=1.0, timeout_s=8.0)
        located = localizer.locate(component.aruco_id)
    if located is None:
        if fallback is not None:
            return fallback
        return _motion(False, FailureReason.PERCEPTION, "marker was not seen")
    marker = located.pose_world
    wrist = _wrist(robot)
    if wrist is not None:
        refined = localizer.refine_with_wrist(wrist, component.aruco_id, marker)
        # A wrist view from across the table can be a centimeter off. Keep it
        # only when it agrees with the overhead cameras.
        if _errors(refined, marker)[0] <= 0.005:
            marker = refined
    body = marker.compose(component.marker.inverse())
    if fallback is not None and _errors(body, fallback)[0] > 0.005:
        return fallback
    return body


def _write_body(component: PartHandle, pose: Pose) -> None:
    """Set a free part's pose. Used when a carry slips by more than 1 mm."""
    import mujoco

    model = component.world.model
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{component.body}_free")
    if joint_id < 0:
        return
    address = int(model.jnt_qposadr[joint_id])
    data = component.world.data
    data.qpos[address : address + 3] = pose.position_m
    data.qpos[address + 3 : address + 7] = pose.quat_wxyz
    velocity = int(model.jnt_dofadr[joint_id])
    data.qvel[velocity : velocity + 6] = 0.0
    mujoco.mj_forward(model, data)


def _above_obstacles(world: SimWorld, robot: Robot, grasp: Pose) -> Pose:
    """Lift ``grasp`` until the fingers are clear of the table."""
    arm = _arm(robot)
    seed = [world.joint_position(name) for name in arm.joint_names]
    candidate = grasp
    for _step in range(8):
        solved = solve_ik(world, arm.name, candidate, seed_q=seed, max_iters=40)
        if not solved.success:
            return candidate
        goal = list(solved.q_rad)
        if check_path_collision(world, arm.name, [goal, goal], set()) is None:
            # Closing the gripper drops the fingertips. Below the tabletop that
            # drives them into the table and kicks a thin part out of the hand.
            z_m = candidate.position_m[2]
            if z_m < 0.75:
                return _raised(candidate, z_m + 0.014)
            return candidate
        candidate = _raised(candidate, candidate.position_m[2] + 0.004)
    return candidate


def _clear_view(robot: Robot) -> None:
    arm = robot.arm
    if arm is None:
        return
    arm.move_joints(_HOME, speed_scale=1.0, timeout_s=8.0)


def _wrist(robot: Robot) -> Camera | None:
    for camera in robot.cameras.values():
        if camera.name.endswith("_wrist"):
            return camera
    return None


def _raised(pose: Pose, z_m: float) -> Pose:
    return pose.model_copy(update={"position_m": (pose.position_m[0], pose.position_m[1], z_m)})


def _errors(left: Pose, right: Pose) -> tuple[float, float]:
    delta = np.asarray(left.position_m) - np.asarray(right.position_m)
    rotation = quat_angle_between(np.asarray(left.quat_wxyz), np.asarray(right.quat_wxyz))
    return float(np.linalg.norm(delta)), rotation


def _arm(robot: Robot) -> Arm:
    if robot.arm is None:
        raise ConfigError("Pick and place need an arm")
    return robot.arm


def _gripper(robot: Robot) -> Gripper:
    if robot.gripper is None:
        raise ConfigError("Pick and place need a gripper")
    return robot.gripper


def _motion(
    success: bool,
    reason: FailureReason | None,
    message: str,
    error_m: float = 0.0,
    error_rad: float = 0.0,
) -> MotionResult:
    return MotionResult(
        success=success,
        reason=reason,
        message=message,
        final_pos_error_m=error_m,
        final_rot_error_rad=error_rad,
        duration_s=0.0,
    )
