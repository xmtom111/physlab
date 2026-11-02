"""Drive the procedural diff-drive base and save an MP4."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from physlab.core.config import load_config
from physlab.core.frames import matrix_to_quat
from physlab.core.logging import configure_logging, get_logger
from physlab.core.types import Pose, Pose2D
from physlab.hal.factory import create_robot
from physlab.robots.catalog import RobotCatalog
from physlab.sim.recorder import Recorder
from physlab.sim.scene import CameraSpec, RobotPlacement, SceneSpec
from physlab.sim.world import SimWorld


def _look_at(eye: tuple[float, float, float], target: tuple[float, float, float]) -> Pose:
    """Camera pose that looks from ``eye`` toward ``target``.

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
    """Drive to (1.0 m, 0.5 m, 1.57 rad) and write the overhead video.

    Raises:
        SystemExit: The base did not reach the pose.
    """
    configure_logging("INFO", False)
    log = get_logger("examples.03b_sim_mobile_base")
    loaded = load_config()
    config = loaded.model_copy(update={"sim": loaded.sim.model_copy(update={"timestep_s": 0.01})})
    camera = CameraSpec(
        name="base_view",
        pose=_look_at((2.2, -1.6, 1.8), (0.6, 0.2, 0.1)),
        fovy_deg=50.0,
        width=320,
        height=240,
    )
    scene = SceneSpec(
        floor_size_m=(8.0, 8.0),
        robots=[
            RobotPlacement(
                robot_id="generic_diffdrive",
                instance="base",
                base_pose=Pose(position_m=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
            )
        ],
        cameras=[camera],
    )
    catalog = RobotCatalog(config.configs_dir, config.assets_dir)
    world = SimWorld(scene, catalog, config)
    robot = create_robot(world, "base")
    if robot.base is None:
        raise SystemExit(1)
    video = _drive(world)
    pose = robot.base.get_pose2d()
    log.info(
        "arrived",
        path=str(video),
        bytes=video.stat().st_size,
        x_m=pose.x_m,
        y_m=pose.y_m,
        yaw_rad=pose.yaw_rad,
    )


def _drive(world: SimWorld) -> Path:
    robot = create_robot(world, "base")
    if robot.base is None:
        raise SystemExit(1)
    step = world.step
    with Recorder(world, "base_view", fps=20.0) as recorder:

        def stepping(n: int = 1) -> None:
            step(n)
            recorder.capture()

        world.step = stepping  # type: ignore[method-assign]
        result = robot.base.move_to(Pose2D(x_m=1.0, y_m=0.5, yaw_rad=1.57), timeout_s=30.0)
        if not result.success:
            raise SystemExit(1)
        return recorder.path


if __name__ == "__main__":
    main()
