"""Mock-print a lens holder and pick it up with an xArm7."""

from __future__ import annotations

import mujoco
from structlog.typing import FilteringBoundLogger

from physlab.core.config import PhysLabConfig, load_config
from physlab.core.logging import configure_logging, get_logger
from physlab.core.types import Pose
from physlab.fabrication.mesh_check import load_printer_profile
from physlab.fabrication.mock_print import MockPrintService
from physlab.fabrication.print_service import fabricate
from physlab.hal.factory import create_robot
from physlab.hal.interfaces import Arm, Gripper
from physlab.robots.catalog import RobotCatalog
from physlab.sim.scene import RobotPlacement, SceneSpec, TableSpec
from physlab.sim.world import SimWorld


def main() -> None:
    """Print the default lens holder into the pickup zone and lift it.

    Raises:
        SystemExit: Menagerie is missing, the print fails, or the pick misses.
    """
    configure_logging("INFO", False)
    log = get_logger("examples.06_print_and_pick")
    loaded = load_config()
    if not (loaded.assets_dir / "mujoco_menagerie" / ".git").is_dir():
        log.info("assets_missing", hint="physlab assets fetch")
        raise SystemExit(1)
    config = loaded.model_copy(update={"sim": loaded.sim.model_copy(update={"timestep_s": 0.002})})
    world = _world(config)
    robot = create_robot(world, "arm")
    if robot.arm is None or robot.gripper is None:
        raise SystemExit(1)
    _pick(world, robot.arm, robot.gripper, log)


def _pick(world: SimWorld, arm: Arm, gripper: Gripper, log: FilteringBoundLogger) -> None:
    """Spawn the lens holder and lift it off the table.

    Args:
        world: Scene that receives the printed part.
        arm: xArm used for the pick.
        gripper: Parallel gripper.
        log: Example logger.

    Raises:
        SystemExit: The print or the pick did not succeed.
    """
    slot = Pose(position_m=(0.48, 0.0, 0.75), quat_wxyz=(1.0, 0.0, 0.0, 0.0))
    service = MockPrintService(world, mm3_per_s=1.0e9, pickup_poses=[slot])
    result = fabricate("lens_holder", {}, service, load_printer_profile())
    pose = None if result.inventory is None else result.inventory.pose
    if not result.ok or result.part is None or pose is None or result.inventory is None:
        raise SystemExit(1)
    grasp = pose.compose(result.part.frames["grasp"])
    if not arm.move_pose(grasp, speed_scale=1.0, timeout_s=12.0).success:
        raise SystemExit(1)
    if not gripper.close().success:
        raise SystemExit(1)
    lifted = grasp.model_copy(
        update={
            "position_m": (grasp.position_m[0], grasp.position_m[1], grasp.position_m[2] + 0.08)
        }
    )
    if not arm.move_pose(lifted, speed_scale=1.0, timeout_s=12.0).success:
        raise SystemExit(1)
    instance = result.inventory.part_instance_id
    body = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, instance)
    height_m = float(world.data.xpos[body][2])
    log.info("picked", instance=instance, z_m=height_m)
    if height_m < 0.78:
        raise SystemExit(1)


def _world(config: PhysLabConfig) -> SimWorld:
    scene = SceneSpec(
        table=TableSpec(),
        robots=[
            RobotPlacement(
                robot_id="ufactory_xarm7",
                instance="arm",
                base_pose=Pose(position_m=(0.0, 0.0, 0.75), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
            )
        ],
    )
    return SimWorld(scene, RobotCatalog(config.configs_dir, config.assets_dir), config)


if __name__ == "__main__":
    main()
