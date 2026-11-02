"""Inverse kinematics, planning, and transit skills."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from physlab.core.config import PhysLabConfig, load_config
from physlab.core.types import Pose
from physlab.hal.factory import create_robot
from physlab.motion.ik import solve_ik
from physlab.motion.indexing import (
    arm_joint_names,
    end_effector_site,
    joint_bounds,
    qpos_addresses,
    write_arm,
)
from physlab.motion.planner import check_path_collision, plan_cartesian_line, plan_joint_path
from physlab.motion.skills import safe_transit
from physlab.motion.trajectory import time_parameterize
from physlab.robots.catalog import RobotCatalog
from physlab.sim.scene import (
    ObjectSpec,
    PrimitiveKind,
    PrimitiveSpec,
    RobotPlacement,
    SceneSpec,
    TableSpec,
)
from physlab.sim.world import SimWorld

_ROBOTS = ("ufactory_xarm7", "franka_emika_panda", "trs_so_arm100")


def _config(tmp_path, seed: int = 0) -> PhysLabConfig:
    loaded = load_config()
    return loaded.model_copy(
        update={
            "seed": seed,
            "run_dir": tmp_path,
            "sim": loaded.sim.model_copy(update={"timestep_s": 0.002}),
        }
    )


def _catalog() -> RobotCatalog:
    loaded = load_config()
    return RobotCatalog(loaded.configs_dir, loaded.assets_dir)


def _arm_world(tmp_path, robot_id: str, *, seed: int = 0) -> SimWorld:
    return SimWorld(
        SceneSpec(
            robots=[
                RobotPlacement(
                    robot_id=robot_id,
                    instance="arm",
                    base_pose=Pose(position_m=(0.0, 0.0, 0.8), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
                )
            ]
        ),
        _catalog(),
        _config(tmp_path, seed),
    )


def _site_pose(world: SimWorld, data: mujoco.MjData, site: str) -> Pose:
    site_id = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_SITE, site)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    matrix[:3, 3] = np.asarray(data.site_xpos[site_id], dtype=np.float64)
    return Pose.from_matrix(matrix)


@pytest.mark.assets
@pytest.mark.parametrize("robot_id", _ROBOTS)
def test_ik_reaches_random_fk_poses(tmp_path, robot_id: str) -> None:
    world = _arm_world(tmp_path, robot_id)
    names = arm_joint_names(world, "arm")
    site = end_effector_site(world, "arm")
    addresses = qpos_addresses(world.model, names)
    lower, upper = joint_bounds(world.model, names)
    rng = np.random.default_rng(0)
    successes = 0
    for _ in range(30):
        sample = rng.uniform(lower, upper)
        data = world.copy_data()
        data.qpos[:] = write_arm(data.qpos, addresses, sample.tolist())
        mujoco.mj_forward(world.model, data)
        target = _site_pose(world, data, site)
        solved = solve_ik(world, "arm", target, max_iters=40)
        successes += int(solved.success)
    assert successes / 30 >= 0.95


def test_time_parameterize_hits_waypoints() -> None:
    trajectory = time_parameterize(
        [[0.0, 0.0], [1.0, -0.5], [1.0, 0.5]],
        [1.0, 1.0],
        [2.0, 2.0],
        1.0,
    )
    assert trajectory.sample(0.0).tolist() == pytest.approx([0.0, 0.0])
    assert trajectory.sample(trajectory.duration_s).tolist() == pytest.approx([1.0, 0.5])
    mid = trajectory.sample(trajectory.times_s[1])
    assert mid.tolist() == pytest.approx([1.0, -0.5], abs=1e-6)


@pytest.mark.assets
def test_cartesian_line_stays_within_2mm(tmp_path) -> None:
    world = _arm_world(tmp_path, "ufactory_xarm7")
    site = end_effector_site(world, "arm")
    start = _site_pose(world, world.copy_data(), site)
    end = Pose(
        position_m=(start.position_m[0] + 0.06, start.position_m[1] - 0.02, start.position_m[2]),
        quat_wxyz=start.quat_wxyz,
    )
    waypoints = plan_cartesian_line(world, "arm", start, end, step_m=0.005)
    names = arm_joint_names(world, "arm")
    addresses = qpos_addresses(world.model, names)
    data = world.copy_data()
    origin = np.asarray(start.position_m, dtype=np.float64)
    dest = np.asarray(end.position_m, dtype=np.float64)
    span = dest - origin
    worst = 0.0
    for joints in waypoints:
        data.qpos[:] = write_arm(data.qpos, addresses, joints)
        mujoco.mj_forward(world.model, data)
        point = _site_pose(world, data, site)
        position = np.asarray(point.position_m, dtype=np.float64)
        fraction = float(np.clip(np.dot(position - origin, span) / np.dot(span, span), 0.0, 1.0))
        distance = float(np.linalg.norm(position - (origin + fraction * span)))
        worst = max(worst, distance)
    assert worst <= 0.002


@pytest.mark.assets
def test_rrt_goes_around_a_wall(tmp_path) -> None:
    wall = ObjectSpec(
        name="wall",
        collision=[PrimitiveSpec(type=PrimitiveKind.BOX, size_m=(0.06, 0.05, 0.08))],
        pose=Pose(position_m=(0.35, 0.19, 1.27), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        mass_kg=1.0,
        free=False,
    )
    world = SimWorld(
        SceneSpec(
            table=TableSpec(),
            robots=[
                RobotPlacement(
                    robot_id="ufactory_xarm7",
                    instance="arm",
                    base_pose=Pose(position_m=(0.0, 0.0, 0.75), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
                )
            ],
            objects=[wall],
        ),
        _catalog(),
        _config(tmp_path, seed=2),
    )
    arm = create_robot(world, "arm").arm
    assert arm is not None
    home = list(arm.get_joint_state().positions_rad)
    goal = home.copy()
    goal[0] = 1.0
    allowed = ["table", "floor"]
    assert check_path_collision(world, "arm", [home, goal], allowed) is not None
    path = plan_joint_path(world, "arm", home, goal, allowed_bodies=allowed)
    assert check_path_collision(world, "arm", path, allowed) is None
    assert len(path) >= 2


@pytest.mark.assets
def test_safe_transit_stays_above_clearance(tmp_path) -> None:
    world = _arm_world(tmp_path, "ufactory_xarm7", seed=1)
    robot = create_robot(world, "arm")
    assert robot.arm is not None
    start = robot.arm.get_ee_pose()
    clearance = start.position_m[2] + 0.04
    target = Pose(
        position_m=(start.position_m[0] - 0.08, start.position_m[1] + 0.05, clearance),
        quat_wxyz=start.quat_wxyz,
    )
    heights: list[float] = []
    step = world.step

    def wrapped(n: int = 1) -> None:
        step(n)
        heights.append(robot.arm.get_ee_pose().position_m[2])

    world.step = wrapped  # type: ignore[method-assign]
    result = safe_transit(robot, target, clearance)
    assert result.success, result.message
    assert heights
    lifted = next(index for index, height in enumerate(heights) if height >= clearance - 1e-3)
    assert min(heights[lifted:]) >= clearance - 1e-3


@pytest.mark.assets
def test_move_pose_reaches_a_nearby_target(tmp_path) -> None:
    world = _arm_world(tmp_path, "ufactory_xarm7")
    robot = create_robot(world, "arm")
    assert robot.arm is not None
    start = robot.arm.get_ee_pose()
    target = Pose(
        position_m=(start.position_m[0] + 0.04, start.position_m[1], start.position_m[2] - 0.02),
        quat_wxyz=start.quat_wxyz,
    )
    result = robot.arm.move_pose(target, speed_scale=1.0, timeout_s=12.0)
    assert result.success, result.message
    assert result.final_pos_error_m < 0.005
    assert result.final_rot_error_rad < 0.05
