"""Differential inverse kinematics on a copy of the simulation state.

The solver is mink's quadratic program: a :class:`mink.FrameTask` on the
end-effector site, a :class:`mink.ConfigurationLimit`, and, when requested, a
:class:`mink.CollisionAvoidanceLimit`. The live ``MjData`` is not written.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import mink
import mujoco
import numpy as np
from pydantic import BaseModel, ConfigDict

from physlab.core.errors import ConfigError
from physlab.core.logging import get_logger
from physlab.core.types import Pose
from physlab.motion.indexing import (
    arm_joint_names,
    end_effector_site,
    frozen_dofs,
    joint_bounds,
    qpos_addresses,
    read_arm,
    write_arm,
)
from physlab.sim.world import SimWorld

_logger = get_logger(__name__)

_SOLVER = "daqp"
_DT = 0.01
_DAMPING = 1e-3
_RESTARTS = 5


class IKResult(BaseModel):
    """Outcome of one inverse-kinematics query.

    Attributes:
        success: Whether the pose error is inside the requested tolerances.
        q_rad: Arm joint positions of the best attempt.
        pos_error_m: Translation error of that attempt, meters.
        rot_error_rad: Rotation error of that attempt, radians.
        iters: Solver iterations used by that attempt.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    success: bool
    q_rad: tuple[float, ...]
    pos_error_m: float
    rot_error_rad: float
    iters: int


def solve_ik(
    world: SimWorld,
    robot_instance: str,
    target: Pose,
    *,
    seed_q: Sequence[float] | None = None,
    pos_tol_m: float = 1e-3,
    rot_tol_rad: float = 1e-2,
    max_iters: int = 300,
    position_only: bool = False,
    avoid_collisions: bool = False,
) -> IKResult:
    """Solve inverse kinematics for ``robot_instance`` on a copied configuration.

    The first attempt starts at ``seed_q`` or the current arm position. On
    failure, up to five more attempts draw joint seeds from a generator seeded
    with ``world.config.seed``.

    Args:
        world: Simulation world. Its live data is only read.
        robot_instance: Placement prefix.
        target: Desired end-effector pose in the world frame.
        seed_q: Arm joints for the first attempt.
        pos_tol_m: Acceptable translation error, meters.
        rot_tol_rad: Acceptable rotation error, radians. Ignored when
            ``position_only`` is true.
        max_iters: Iterations per attempt.
        position_only: Track position and leave orientation free.
        avoid_collisions: Add a mink collision-avoidance limit between the
            robot and the rest of the scene.

    Returns:
        The best attempt. ``success`` is false when every attempt misses the
        tolerances.

    Raises:
        ConfigError: ``seed_q`` does not match the arm dof.
    """
    return _search(
        world,
        robot_instance,
        target,
        seed_q,
        pos_tol_m,
        rot_tol_rad,
        max_iters,
        position_only,
        avoid_collisions,
    )


def _search(
    world: SimWorld,
    robot_instance: str,
    target: Pose,
    seed_q: Sequence[float] | None,
    pos_tol_m: float,
    rot_tol_rad: float,
    max_iters: int,
    position_only: bool,
    avoid_collisions: bool,
) -> IKResult:
    names = arm_joint_names(world, robot_instance)
    site = end_effector_site(world, robot_instance)
    addresses = qpos_addresses(world.model, names)
    cloned = world.copy_data()
    seeds = _seeds(world, names, read_arm(cloned.qpos, addresses), seed_q)
    best: IKResult | None = None
    for seed in seeds:
        attempt = _attempt(
            world,
            names,
            site,
            addresses,
            np.array(cloned.qpos, dtype=np.float64, copy=True),
            seed,
            target,
            pos_tol_m,
            rot_tol_rad,
            max_iters,
            position_only,
            avoid_collisions,
        )
        if best is None or attempt.pos_error_m < best.pos_error_m:
            best = attempt
        if attempt.success:
            return attempt
    assert best is not None
    return best


def _seeds(
    world: SimWorld,
    names: tuple[str, ...],
    current: list[float],
    seed_q: Sequence[float] | None,
) -> list[list[float]]:
    seeds = [current] if seed_q is None else [_as_seed(seed_q, len(names))]
    lower, upper = joint_bounds(world.model, names)
    rng = np.random.default_rng(world.config.seed)
    seeds.extend(rng.uniform(lower, upper).tolist() for _ in range(_RESTARTS))
    return seeds


def _as_seed(seed_q: Sequence[float], dof: int) -> list[float]:
    values = [float(item) for item in seed_q]
    if len(values) != dof:
        raise ConfigError(
            "seed_q length must match the arm dof",
            details={"dof": dof, "length": len(values)},
        )
    return values


def _attempt(
    world: SimWorld,
    names: tuple[str, ...],
    site: str,
    addresses: tuple[int, ...],
    qpos: np.ndarray,
    seed: list[float],
    target: Pose,
    pos_tol_m: float,
    rot_tol_rad: float,
    max_iters: int,
    position_only: bool,
    avoid_collisions: bool,
) -> IKResult:
    configuration = mink.Configuration(world.model)
    configuration.update(write_arm(qpos, addresses, seed))
    task = mink.FrameTask(
        frame_name=site,
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.0 if position_only else 1.0,
        lm_damping=1e-3,
    )
    task.set_target(mink.SE3.from_matrix(target.to_matrix()))
    limits: list[mink.Limit] = [mink.ConfigurationLimit(world.model)]
    collision = _collision_limit(world, names[0].split("/", 1)[0]) if avoid_collisions else None
    if collision is not None:
        limits.append(collision)
    frozen = frozen_dofs(world.model, names)
    constraints = [mink.DofFreezingTask(world.model, frozen)] if frozen else None
    return _iterate(
        configuration,
        task,
        limits,
        constraints,
        addresses,
        site,
        target,
        pos_tol_m,
        rot_tol_rad,
        max_iters,
        position_only,
    )


def _iterate(
    configuration: mink.Configuration,
    task: mink.FrameTask,
    limits: list[mink.Limit],
    constraints: list[mink.DofFreezingTask] | None,
    addresses: tuple[int, ...],
    site: str,
    target: Pose,
    pos_tol_m: float,
    rot_tol_rad: float,
    max_iters: int,
    position_only: bool,
) -> IKResult:
    best_q = read_arm(configuration.q, addresses)
    best_pos, best_rot = _pose_error(configuration, site, target)
    used = 0
    for index in range(max_iters):
        used = index + 1
        if _converged(best_pos, best_rot, pos_tol_m, rot_tol_rad, position_only):
            break
        try:
            velocity = mink.solve_ik(
                configuration,
                [task],
                _DT,
                _SOLVER,
                damping=_DAMPING,
                limits=limits,
                constraints=constraints,
            )
        except mink.NoSolutionFound:
            _logger.debug("ik_no_solution", iters=used)
            break
        configuration.integrate_inplace(velocity, _DT)
        best_q = read_arm(configuration.q, addresses)
        best_pos, best_rot = _pose_error(configuration, site, target)
    success = _converged(best_pos, best_rot, pos_tol_m, rot_tol_rad, position_only)
    return IKResult(
        success=success,
        q_rad=tuple(best_q),
        pos_error_m=best_pos,
        rot_error_rad=best_rot,
        iters=used,
    )


def _converged(
    pos_error_m: float,
    rot_error_rad: float,
    pos_tol_m: float,
    rot_tol_rad: float,
    position_only: bool,
) -> bool:
    if pos_error_m > pos_tol_m:
        return False
    return position_only or rot_error_rad <= rot_tol_rad


def _pose_error(configuration: mink.Configuration, site: str, target: Pose) -> tuple[float, float]:
    transform = configuration.get_transform_frame_to_world(site, "site")
    position = transform.translation()
    quat = transform.rotation().wxyz
    current = Pose(
        position_m=(float(position[0]), float(position[1]), float(position[2])),
        quat_wxyz=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
    )
    return current.error_to(target)


def _collision_limit(world: SimWorld, instance: str) -> mink.CollisionAvoidanceLimit | None:
    robot, other = _geom_groups(world.model, instance)
    if not robot or not other:
        return None
    return mink.CollisionAvoidanceLimit(
        world.model,
        [(robot, other)],
        minimum_distance_from_collisions=0.0,
        collision_detection_distance=0.02,
    )


def _geom_groups(model: mujoco.MjModel, instance: str) -> tuple[list[str], list[str]]:
    robot: list[str] = []
    other: list[str] = []
    prefix = f"{instance}/"
    anchor = f"{instance}_"
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if not name:
            continue
        body_id = int(model.geom_bodyid[geom_id])
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body.startswith(prefix) or body.startswith(anchor):
            robot.append(name)
        else:
            other.append(name)
    return robot, other
