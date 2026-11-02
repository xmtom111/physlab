"""Joint-space and Cartesian planners.

``plan_joint_path`` returns the straight line when it is free, otherwise an
RRT-Connect path in joint space. The tree search, including shortcut
smoothing, is the block below ``_rrt_connect`` and stays under 250 lines.
Collision checks run on a copied ``MjData``.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from physlab.core.errors import PlanningFailed
from physlab.core.frames import quat_wxyz_to_xyzw, quat_xyzw_to_wxyz
from physlab.core.types import Pose
from physlab.motion.ik import solve_ik
from physlab.motion.indexing import arm_joint_names, joint_bounds, qpos_addresses, read_arm
from physlab.sim.world import SimWorld

_DEG = math.pi / 180.0
_CONTACT_DIST = 1e-4
_MAX_SAMPLES = 5000
_STEP_RAD = 0.35
_JUMP_RAD = 0.3
_Tree = list[tuple[list[float], int]]


def check_path_collision(
    world: SimWorld,
    instance: str,
    q_path: Sequence[Sequence[float]],
    allowed_bodies: Collection[str],
) -> int | None:
    """Return the first waypoint index whose segment collides.

    Intermediate samples keep the largest joint step at or below one degree.
    A contact counts when a robot body touches a body that is not in
    ``allowed_bodies``.

    Args:
        world: Simulation world. Live data is not modified.
        instance: Robot prefix.
        q_path: Arm joint waypoints.
        allowed_bodies: Bodies the robot may touch.

    Returns:
        Index into ``q_path``, or ``None`` when the path is free.
    """
    if not q_path:
        return None
    names = arm_joint_names(world, instance)
    addresses = qpos_addresses(world.model, names)
    data = world.copy_data()
    allowed = set(allowed_bodies)
    dense, owners = _densify(q_path)
    for sample, owner in zip(dense, owners, strict=True):
        _apply(data, addresses, sample)
        mujoco.mj_forward(world.model, data)
        if _hits(world.model, data, instance, allowed):
            return owner
    return None


def plan_joint_path(
    world: SimWorld,
    instance: str,
    start_q: Sequence[float],
    goal_q: Sequence[float],
    *,
    allowed_bodies: Sequence[str] = (),
) -> list[list[float]]:
    """Plan an arm path from ``start_q`` to ``goal_q``.

    Args:
        world: Simulation used for collision checks.
        instance: Robot prefix.
        start_q: Starting arm joints.
        goal_q: Goal arm joints.
        allowed_bodies: Extra bodies the robot may touch. Bodies already in
            contact at ``start_q`` are included as well.

    Returns:
        Joint waypoints, including the start and the goal.

    Raises:
        PlanningFailed: The straight line and RRT-Connect both fail.
    """
    start = [float(value) for value in start_q]
    goal = [float(value) for value in goal_q]
    allowed = set(allowed_bodies) | _touching(world, instance, start)
    if check_path_collision(world, instance, [start, goal], allowed) is None:
        return [start, goal]
    rng = np.random.default_rng(world.config.seed)
    path = _rrt_connect(world, instance, start, goal, allowed, rng)
    return _shortcut(world, instance, path, allowed, rng)


def plan_cartesian_line(
    world: SimWorld,
    instance: str,
    start: Pose,
    end: Pose,
    step_m: float = 0.005,
) -> list[list[float]]:
    """Follow a straight Cartesian segment with successive inverse kinematics.

    Args:
        world: Simulation whose current arm position seeds the first solve.
        instance: Robot prefix.
        start: Pose at the beginning of the segment.
        end: Pose at the end of the segment.
        step_m: Maximum spacing along the translation, meters.

    Returns:
        Arm joint waypoints.

    Raises:
        PlanningFailed: A step misses its pose or jumps by more than 0.3 rad.
    """
    if step_m <= 0.0:
        raise PlanningFailed("Cartesian step must be positive", details={"step_m": step_m})
    count = max(1, math.ceil(_distance(start, end) / step_m))
    names = arm_joint_names(world, instance)
    addresses = qpos_addresses(world.model, names)
    previous = read_arm(world.copy_data().qpos, addresses)
    waypoints: list[list[float]] = []
    for index in range(count + 1):
        pose = _interpolate(start, end, index / count)
        solved = solve_ik(world, instance, pose, seed_q=previous, max_iters=80)
        if not solved.success:
            raise PlanningFailed(
                "Cartesian IK failed",
                details={"index": index, "pos_error_m": solved.pos_error_m},
            )
        current = list(solved.q_rad)
        jump = max(abs(a - b) for a, b in zip(current, previous, strict=True))
        if waypoints and jump > _JUMP_RAD:
            raise PlanningFailed("Cartesian joint jump", details={"jump_rad": jump})
        waypoints.append(current)
        previous = current
    del names
    return waypoints


def _densify(q_path: Sequence[Sequence[float]]) -> tuple[list[list[float]], list[int]]:
    dense = [[float(value) for value in q_path[0]]]
    owners = [0]
    for index in range(1, len(q_path)):
        start = np.asarray(q_path[index - 1], dtype=np.float64)
        end = np.asarray(q_path[index], dtype=np.float64)
        steps = max(1, math.ceil(float(np.max(np.abs(end - start))) / _DEG))
        for step in range(1, steps + 1):
            dense.append((start + (end - start) * (step / steps)).tolist())
            owners.append(index)
    return dense, owners


def _apply(data: mujoco.MjData, addresses: tuple[int, ...], arm_q: Sequence[float]) -> None:
    for address, value in zip(addresses, arm_q, strict=True):
        data.qpos[address] = value


def _hits(model: mujoco.MjModel, data: mujoco.MjData, instance: str, allowed: set[str]) -> bool:
    prefix = f"{instance}/"
    anchor = f"{instance}_"
    for index in range(data.ncon):
        contact = data.contact[index]
        if float(contact.dist) > _CONTACT_DIST:
            continue
        body_a = _body_name(model, int(model.geom_bodyid[int(contact.geom1)]))
        body_b = _body_name(model, int(model.geom_bodyid[int(contact.geom2)]))
        other = _external(body_a, body_b, prefix, anchor)
        if other is not None and other not in allowed:
            return True
    return False


def _body_name(model: mujoco.MjModel, body_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    return "" if name is None else str(name)


def _external(body_a: str, body_b: str, prefix: str, anchor: str) -> str | None:
    robot_a = body_a.startswith(prefix) or body_a.startswith(anchor)
    robot_b = body_b.startswith(prefix) or body_b.startswith(anchor)
    if robot_a and not robot_b:
        return body_b
    if robot_b and not robot_a:
        return body_a
    return None


def _touching(world: SimWorld, instance: str, arm_q: list[float]) -> set[str]:
    names = arm_joint_names(world, instance)
    addresses = qpos_addresses(world.model, names)
    data = world.copy_data()
    _apply(data, addresses, arm_q)
    mujoco.mj_forward(world.model, data)
    found: set[str] = set()
    prefix = f"{instance}/"
    anchor = f"{instance}_"
    for index in range(data.ncon):
        contact = data.contact[index]
        body_a = _body_name(world.model, int(world.model.geom_bodyid[int(contact.geom1)]))
        body_b = _body_name(world.model, int(world.model.geom_bodyid[int(contact.geom2)]))
        other = _external(body_a, body_b, prefix, anchor)
        if other:
            found.add(other)
    return found


def _rrt_connect(
    world: SimWorld,
    instance: str,
    start: list[float],
    goal: list[float],
    allowed: set[str],
    rng: np.random.Generator,
) -> list[list[float]]:
    """RRT-Connect in joint space. This function and its helpers are the planner."""
    names = arm_joint_names(world, instance)
    lower, upper = joint_bounds(world.model, names)
    tree_a: _Tree = [(start, -1)]
    tree_b: _Tree = [(goal, -1)]
    for _sample in range(_MAX_SAMPLES):
        other = tree_b[0][0]
        sample = list(other) if rng.random() < 0.15 else rng.uniform(lower, upper).tolist()
        grown = _extend(world, instance, tree_a, sample, allowed)
        if grown is not None and _connect(world, instance, tree_b, tree_a[grown][0], allowed):
            return _join(tree_a, grown, tree_b, start)
        tree_a, tree_b = tree_b, tree_a
    raise PlanningFailed(
        "RRT-Connect did not find a joint path",
        details={"samples": _MAX_SAMPLES},
    )


def _extend(
    world: SimWorld,
    instance: str,
    tree: _Tree,
    target: list[float],
    allowed: set[str],
) -> int | None:
    nearest = _nearest(tree, target)
    parent = nearest
    current = tree[nearest][0]
    reached: int | None = None
    for _ in range(8):
        nxt = _steer(current, target)
        if check_path_collision(world, instance, [current, nxt], allowed) is not None:
            break
        tree.append((nxt, parent))
        parent = len(tree) - 1
        current = nxt
        reached = parent
        if _close(current, target):
            break
    return reached


def _connect(
    world: SimWorld,
    instance: str,
    tree: _Tree,
    target: list[float],
    allowed: set[str],
) -> bool:
    guard = 0
    while guard < 30 and not _close(tree[_nearest(tree, target)][0], target):
        guard += 1
        before = len(tree)
        grown = _extend(world, instance, tree, target, allowed)
        if grown is None or len(tree) == before:
            return False
    return _close(tree[_nearest(tree, target)][0], target)


def _join(tree_a: _Tree, index_a: int, tree_b: _Tree, start: list[float]) -> list[list[float]]:
    path_a = _chain(tree_a, index_a)
    path_b = _chain(tree_b, _nearest(tree_b, path_a[-1]))
    if _close(tree_a[0][0], start):
        return path_a[:-1] + path_b[::-1]
    return path_b[:-1] + path_a[::-1]


def _chain(tree: _Tree, index: int) -> list[list[float]]:
    path: list[list[float]] = []
    while index >= 0:
        path.append(tree[index][0])
        index = tree[index][1]
    path.reverse()
    return path


def _nearest(tree: _Tree, target: list[float]) -> int:
    target_v = np.asarray(target, dtype=np.float64)
    best = 0
    best_distance = float("inf")
    for index, (node, _parent) in enumerate(tree):
        distance = float(np.linalg.norm(np.asarray(node) - target_v))
        if distance < best_distance:
            best = index
            best_distance = distance
    return best


def _steer(start: list[float], goal: list[float]) -> list[float]:
    delta = np.asarray(goal, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    span = float(np.max(np.abs(delta)))
    if span <= _STEP_RAD or span == 0.0:
        return list(goal)
    stepped = np.asarray(start, dtype=np.float64) + delta * (_STEP_RAD / span)
    return [float(value) for value in stepped]


def _close(left: list[float], right: list[float]) -> bool:
    return max(abs(a - b) for a, b in zip(left, right, strict=True)) < 1e-6


def _shortcut(
    world: SimWorld,
    instance: str,
    path: list[list[float]],
    allowed: set[str],
    rng: np.random.Generator,
) -> list[list[float]]:
    for _ in range(40):
        if len(path) < 3:
            break
        left = int(rng.integers(0, len(path) - 2))
        right = int(rng.integers(left + 2, len(path)))
        if check_path_collision(world, instance, [path[left], path[right]], allowed) is None:
            path = path[: left + 1] + path[right:]
    return path


def _distance(start: Pose, end: Pose) -> float:
    delta = np.asarray(end.position_m) - np.asarray(start.position_m)
    return float(np.linalg.norm(delta))


def _interpolate(start: Pose, end: Pose, fraction: float) -> Pose:
    position = (1.0 - fraction) * np.asarray(start.position_m) + fraction * np.asarray(
        end.position_m
    )
    key = Rotation.from_quat(
        np.vstack(
            [
                quat_wxyz_to_xyzw(np.asarray(start.quat_wxyz)),
                quat_wxyz_to_xyzw(np.asarray(end.quat_wxyz)),
            ]
        )
    )
    quat = quat_xyzw_to_wxyz(Slerp([0.0, 1.0], key)([fraction]).as_quat()[0])
    return Pose(
        position_m=(float(position[0]), float(position[1]), float(position[2])),
        quat_wxyz=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
        frame=start.frame,
    )
