"""Drive a diff-drive SO-ARM100 1.5 m to a table and pick up a cube.

The arm is mounted with ``RobotPlacement.arm_id``. ``navigate_and_reach``
drives the base 1.5 m and puts the end effector on the cube, then the gripper
closes. Seeds 0 through 4 use the same geometry.
"""

from __future__ import annotations

import mujoco

from physlab.core.config import load_config
from physlab.core.logging import configure_logging, get_logger
from physlab.core.types import Pose, Pose2D
from physlab.hal.factory import create_robot
from physlab.motion.skills import navigate_and_reach
from physlab.robots.catalog import RobotCatalog
from physlab.sim.scene import (
    ObjectSpec,
    PrimitiveKind,
    PrimitiveSpec,
    RobotPlacement,
    SceneSpec,
)
from physlab.sim.world import SimWorld


def main() -> None:
    """Run the mobile pick for seeds 0 through 4.

    Raises:
        SystemExit: Menagerie is missing or a seed does not carry the cube.
    """
    configure_logging("INFO", False)
    log = get_logger("examples.04b_mobile_pick")
    loaded = load_config()
    if not (loaded.assets_dir / "mujoco_menagerie" / ".git").is_dir():
        log.info("assets_missing", hint="physlab assets fetch")
        raise SystemExit(1)
    for seed in range(5):
        carried = _pick(loaded, seed)
        log.info("carried", seed=seed, x_m=carried[0], y_m=carried[1], z_m=carried[2])


def _pick(loaded, seed: int) -> tuple[float, float, float]:
    config = loaded.model_copy(
        update={"seed": seed, "sim": loaded.sim.model_copy(update={"timestep_s": 0.002})}
    )
    table = ObjectSpec(
        name="table",
        collision=[PrimitiveSpec(type=PrimitiveKind.BOX, size_m=(0.3, 0.3, 0.24))],
        pose=Pose(position_m=(1.9, 0.55, 0.12), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        mass_kg=4.0,
        free=False,
        rgba=(0.45, 0.32, 0.22, 1.0),
    )
    scene = SceneSpec(
        floor_size_m=(6.0, 6.0),
        robots=[
            RobotPlacement(
                robot_id="generic_diffdrive",
                instance="mm",
                base_pose=Pose(position_m=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
                arm_id="trs_so_arm100",
            )
        ],
        objects=[table],
    )
    catalog = RobotCatalog(config.configs_dir, config.assets_dir)
    world = SimWorld(scene, catalog, config)
    robot = create_robot(world, "mm")
    if robot.arm is None or robot.base is None or robot.gripper is None:
        raise SystemExit(1)
    robot.gripper.open()
    home = robot.arm.get_ee_pose()
    cube_pose = Pose(
        position_m=(home.position_m[0] + 1.5, home.position_m[1], home.position_m[2]),
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    reached = navigate_and_reach(
        robot,
        Pose2D(x_m=1.5, y_m=0.0, yaw_rad=0.0),
        cube_pose.model_copy(update={"quat_wxyz": home.quat_wxyz}),
    )
    if not reached.success:
        raise SystemExit(1)
    arrived = robot.base.get_pose2d()
    if abs(arrived.x_m - 1.5) > 0.02 or abs(arrived.y_m) > 0.02:
        raise SystemExit(1)
    world.add_object_at_runtime(
        ObjectSpec(
            name="cube",
            collision=[PrimitiveSpec(type=PrimitiveKind.BOX, size_m=(0.025, 0.025, 0.025))],
            pose=cube_pose,
            mass_kg=0.03,
            free=True,
            rgba=(0.2, 0.55, 0.85, 1.0),
        )
    )
    if not robot.gripper.close().success:
        raise SystemExit(1)
    hand = robot.arm.get_ee_pose().position_m
    if abs(hand[0] - cube_pose.position_m[0]) > 0.02:
        raise SystemExit(1)
    if abs(hand[1] - cube_pose.position_m[1]) > 0.02:
        raise SystemExit(1)
    body = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    position = world.data.xpos[body]
    return float(position[0]), float(position[1]), float(position[2])


if __name__ == "__main__":
    main()
