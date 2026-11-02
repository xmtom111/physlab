"""Move an xArm7 in joint space and save an MP4."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from physlab.core.config import load_config
from physlab.core.frames import matrix_to_quat
from physlab.core.logging import configure_logging, get_logger
from physlab.core.types import Pose
from physlab.hal.factory import create_robot
from physlab.robots.catalog import RobotCatalog
from physlab.sim.recorder import Recorder
from physlab.sim.scene import (
    CameraSpec,
    ObjectSpec,
    PrimitiveKind,
    PrimitiveSpec,
    RobotPlacement,
    SceneSpec,
    TableSpec,
)
from physlab.sim.world import SimWorld

_HOME = (0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0)


def _look_at(eye: tuple[float, float, float], target: tuple[float, float, float]) -> Pose:
    """Camera pose that looks from ``eye`` toward ``target``.

    MuJoCo cameras look along their negative Z axis.

    Args:
        eye: Camera position, meters.
        target: Point the camera faces, meters.

    Returns:
        Pose in the world frame.
    """
    eye_v = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye_v
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation = np.column_stack((right, up, -forward))
    quat = matrix_to_quat(rotation)
    return Pose(position_m=tuple(float(value) for value in eye_v), quat_wxyz=tuple(quat))


def main() -> None:
    """Place an xArm7 on a table, move three joints, and write a video.

    Raises:
        SystemExit: Menagerie is missing or a joint motion failed.
    """
    configure_logging("INFO", False)
    log = get_logger("examples.03_sim_arm_joints")
    loaded = load_config()
    assets = loaded.assets_dir / "mujoco_menagerie" / ".git"
    if not assets.is_dir():
        log.info("assets_missing", hint="physlab assets fetch")
        raise SystemExit(1)
    config = loaded.model_copy(update={"sim": loaded.sim.model_copy(update={"timestep_s": 0.002})})
    cube = ObjectSpec(
        name="cube",
        collision=[PrimitiveSpec(type=PrimitiveKind.BOX, size_m=(0.04, 0.04, 0.04))],
        pose=Pose(position_m=(0.42, 0.12, 0.79), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        mass_kg=0.05,
        free=True,
        rgba=(0.85, 0.35, 0.2, 1.0),
    )
    camera = CameraSpec(
        name="overview",
        pose=_look_at((1.15, -1.05, 1.45), (0.2, 0.0, 0.9)),
        fovy_deg=45.0,
        width=320,
        height=240,
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
        cameras=[camera],
    )
    catalog = RobotCatalog(config.configs_dir, config.assets_dir)
    world = SimWorld(scene, catalog, config)
    video = _record(world)
    log.info("video", path=str(video), bytes=video.stat().st_size)


def _record(world: SimWorld) -> Path:
    robot = create_robot(world, "arm")
    if robot.arm is None:
        raise SystemExit(1)
    arm = robot.arm
    step = world.step
    with Recorder(world, "overview", fps=20.0) as recorder:

        def stepping(n: int = 1) -> None:
            step(n)
            recorder.capture()

        world.step = stepping  # type: ignore[method-assign]
        held = arm.move_joints(_HOME, speed_scale=1.0, timeout_s=4.0)
        delta = (0.35, -0.15, 0.1, 0.1, 0.0, 0.0, 0.0)
        goal = [value + offset for value, offset in zip(_HOME, delta, strict=True)]
        moved = arm.move_joints(goal, speed_scale=1.0, timeout_s=8.0)
        if not held.success or not moved.success:
            get_logger("examples.03_sim_arm_joints").error(
                "motion_failed",
                held=held.message,
                moved=moved.message,
                held_reason=None if held.reason is None else held.reason.value,
                moved_reason=None if moved.reason is None else moved.reason.value,
            )
            raise SystemExit(1)
        return recorder.path


if __name__ == "__main__":
    main()
