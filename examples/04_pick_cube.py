"""Pick up a cube with an xArm7 and set it down 20 cm away.

The grasp points the tool +Z axis downward. Seeds 0 through 4 share the same
scene; the seed is stored on the config so IK restarts stay reproducible.
"""

from __future__ import annotations

import mujoco

from physlab.core.config import load_config
from physlab.core.logging import configure_logging, get_logger
from physlab.core.types import Pose
from physlab.hal.factory import create_robot
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

_DOWN = (0.0, 1.0, 0.0, 0.0)
_CUBE_X = 0.48
_CUBE_Z = 0.77
_GRASP_Z = 0.745


def main() -> None:
    """Run the pick-and-place for seeds 0 through 4.

    Raises:
        SystemExit: Menagerie is missing or a seed does not move the cube.
    """
    configure_logging("INFO", False)
    log = get_logger("examples.04_pick_cube")
    loaded = load_config()
    if not (loaded.assets_dir / "mujoco_menagerie" / ".git").is_dir():
        log.info("assets_missing", hint="physlab assets fetch")
        raise SystemExit(1)
    for seed in range(5):
        placed = _pick(loaded, seed)
        log.info("placed", seed=seed, x_m=placed[0], y_m=placed[1], z_m=placed[2])


def _pick(loaded, seed: int) -> tuple[float, float, float]:
    config = loaded.model_copy(
        update={"seed": seed, "sim": loaded.sim.model_copy(update={"timestep_s": 0.002})}
    )
    cube = ObjectSpec(
        name="cube",
        collision=[PrimitiveSpec(type=PrimitiveKind.BOX, size_m=(0.04, 0.04, 0.04))],
        pose=Pose(position_m=(_CUBE_X, 0.0, _CUBE_Z), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        mass_kg=0.05,
        free=True,
        rgba=(0.85, 0.35, 0.2, 1.0),
    )
    scene = SceneSpec(
        table=TableSpec(),
        robots=[
            RobotPlacement(
                robot_id="ufactory_xarm7",
                instance="arm",
                base_pose=Pose(position_m=(0.0, 0.0, 0.75), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
            )
        ],
        objects=[cube],
    )
    catalog = RobotCatalog(config.configs_dir, config.assets_dir)
    world = SimWorld(scene, catalog, config)
    robot = create_robot(world, "arm")
    if robot.arm is None or robot.gripper is None:
        raise SystemExit(1)
    grasp = Pose(position_m=(_CUBE_X, 0.0, _GRASP_Z), quat_wxyz=_DOWN)
    if not robot.arm.move_pose(grasp, speed_scale=1.0, timeout_s=12.0).success:
        raise SystemExit(1)
    if not robot.gripper.close().success:
        raise SystemExit(1)
    lifted = Pose(position_m=(_CUBE_X, 0.0, 0.82), quat_wxyz=_DOWN)
    carried = Pose(position_m=(_CUBE_X - 0.20, 0.0, 0.82), quat_wxyz=_DOWN)
    if not robot.arm.move_pose(lifted, speed_scale=1.0, timeout_s=12.0).success:
        raise SystemExit(1)
    if not robot.arm.move_pose(carried, speed_scale=1.0, timeout_s=12.0).success:
        raise SystemExit(1)
    robot.gripper.open()
    body = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    position = world.data.xpos[body]
    if abs(float(position[0]) - (_CUBE_X - 0.20)) > 0.03 or float(position[2]) < 0.7:
        raise SystemExit(1)
    return float(position[0]), float(position[1]), float(position[2])


if __name__ == "__main__":
    main()
