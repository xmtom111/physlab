"""Scene builder, MuJoCo world, and HAL backends."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
from pydantic import ValidationError

from physlab.core.config import PhysLabConfig, load_config
from physlab.core.errors import AssetMissing, ConfigError
from physlab.core.types import FailureReason, Pose, Pose2D
from physlab.hal.factory import create_camera, create_robot
from physlab.hal.interfaces import Arm, Camera, Gripper, MobileBase
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
    build_mjspec,
)
from physlab.sim.world import SimWorld

_HOME = (0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0)


def _pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Pose:
    return Pose(position_m=(x, y, z), quat_wxyz=(1.0, 0.0, 0.0, 0.0))


def _config(tmp_path: Path, timestep_s: float) -> PhysLabConfig:
    loaded = load_config()
    return loaded.model_copy(
        update={
            "run_dir": tmp_path,
            "sim": loaded.sim.model_copy(update={"timestep_s": timestep_s}),
        }
    )


def _catalog() -> RobotCatalog:
    loaded = load_config()
    return RobotCatalog(loaded.configs_dir, loaded.assets_dir)


def _cube(name: str, pose: Pose, *, free: bool = False) -> ObjectSpec:
    return ObjectSpec(
        name=name,
        collision=[PrimitiveSpec(type=PrimitiveKind.BOX, size_m=(0.04, 0.04, 0.04))],
        pose=pose,
        mass_kg=0.05,
        free=free,
    )


def _xarm_scene(*objects: ObjectSpec, camera: CameraSpec | None = None) -> SceneSpec:
    cameras = [] if camera is None else [camera]
    return SceneSpec(
        table=TableSpec(),
        robots=[RobotPlacement(robot_id="ufactory_xarm7", instance="arm", base_pose=_pose(z=0.75))],
        objects=list(objects),
        cameras=cameras,
    )


def _triangle(path: Path) -> None:
    path.write_text(
        "v 0 0 0\nv 0.02 0 0\nv 0 0.02 0\nv 0 0 0.02\nf 1 2 3\nf 1 3 4\nf 1 4 2\nf 2 4 3\n",
        encoding="utf-8",
    )


def test_scene_rejects_empty_collision_and_duplicate_instances(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        ObjectSpec(name="empty", collision=[], pose=_pose(), mass_kg=0.1)
    with pytest.raises(ValidationError):
        CameraSpec(name="cam")
    catalog = _catalog()
    scene = SceneSpec(
        robots=[
            RobotPlacement(robot_id="gantry_xyz", instance="g", base_pose=_pose(z=0.2)),
            RobotPlacement(robot_id="gantry_xyz", instance="g", base_pose=_pose(z=0.4)),
        ]
    )
    with pytest.raises(ConfigError, match="Duplicate"):
        build_mjspec(scene, catalog)
    bad = ObjectSpec(
        name="bad",
        collision=[PrimitiveSpec(type=PrimitiveKind.BOX, size_m=(0.1,))],
        pose=_pose(),
        mass_kg=0.1,
    )
    with pytest.raises(ConfigError, match="Box"):
        build_mjspec(SceneSpec(objects=[bad]), catalog)
    with pytest.raises(AssetMissing):
        missing = _cube("meshobj", _pose(z=0.2)).model_copy(
            update={"visual_mesh": tmp_path / "missing.obj"}
        )
        build_mjspec(SceneSpec(objects=[missing]), catalog)


def test_visual_mesh_is_not_a_collider(tmp_path: Path) -> None:
    mesh = tmp_path / "tetra.obj"
    _triangle(mesh)
    obj = _cube("widget", _pose(z=0.2), free=True).model_copy(
        update={"visual_mesh": mesh, "aruco_id": 4}
    )
    spec = build_mjspec(SceneSpec(objects=[obj]), _catalog())
    model = spec.compile()
    visual = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "widget_visual")
    collision = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "widget_col_0")
    assert visual >= 0 and collision >= 0
    assert int(model.geom_contype[visual]) == 0
    assert int(model.geom_conaffinity[visual]) == 0
    assert int(model.geom_contype[collision]) == 1
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "widget_aruco_4") >= 0
    assert model.geom_size[collision][0] == pytest.approx(0.02)


def test_world_step_reset_contacts_and_copy(tmp_path: Path) -> None:
    catalog = _catalog()
    world = SimWorld(
        SceneSpec(
            objects=[_cube("cube", _pose(0.2, 0.0, 0.1), free=True)],
            robots=[RobotPlacement(robot_id="gantry_xyz", instance="g", base_pose=_pose(z=0.3))],
        ),
        catalog,
        _config(tmp_path, 0.01),
    )
    with pytest.raises(ConfigError):
        world.step(-1)
    assert world.run_until(lambda item: item.time_s >= 0.02, timeout_s=1.0)
    assert world.time_s >= 0.02
    assert world.run_until(lambda _item: False, timeout_s=0.0) is False
    cloned = world.copy_data()
    assert cloned is not world.data
    cloned.qpos[0] = 3.0
    assert float(world.data.qpos[0]) != 3.0
    world.set_joint_position("g/x", 0.2)
    assert world.contacts() == [] or all(item.dist <= 0.05 for item in world.contacts())
    world.reset(seed=3)
    assert world.joint_position("g/x") == pytest.approx(0.0)
    with pytest.raises(ConfigError):
        world.joint_position("missing")


def test_add_object_preserves_joints_and_weld_holds(tmp_path: Path) -> None:
    catalog = _catalog()
    left = _cube("left", _pose(-0.1, 0.0, 0.1), free=True)
    right = _cube("right", _pose(0.1, 0.0, 0.1), free=True)
    world = SimWorld(SceneSpec(objects=[left, right]), catalog, _config(tmp_path, 0.002))
    world.weld("left", "right")
    start = world.body_pose("left").position_m
    other = world.body_pose("right").position_m
    gap = math.dist(start, other)
    world.step(100)
    later = math.dist(world.body_pose("left").position_m, world.body_pose("right").position_m)
    assert later == pytest.approx(gap, abs=1e-3)
    world.unweld("left", "right")
    with pytest.raises(ConfigError, match="not welded"):
        world.unweld("left", "right")
    with pytest.raises(ConfigError, match="already exists"):
        world.add_object_at_runtime(left)
    gantry = SimWorld(
        SceneSpec(
            robots=[RobotPlacement(robot_id="gantry_xyz", instance="g", base_pose=_pose(z=0.2))]
        ),
        catalog,
        _config(tmp_path, 0.002),
    )
    gantry.set_joint_position("g/x", 0.25)
    gantry.set_joint_position("g/y", 0.1)
    saved = (gantry.joint_position("g/x"), gantry.joint_position("g/y"))
    gantry.add_object_at_runtime(_cube("extra", _pose(0.0, 0.4, 0.2), free=True))
    assert gantry.joint_position("g/x") == pytest.approx(saved[0])
    assert gantry.joint_position("g/y") == pytest.approx(saved[1])
    assert mujoco.mj_name2id(gantry.model, mujoco.mjtObj.mjOBJ_JOINT, "extra_free") >= 0


@pytest.mark.assets
def test_xarm_prefixed_and_joint_motion(tmp_path: Path) -> None:
    world = SimWorld(
        _xarm_scene(_cube("cube", _pose(0.45, 0.1, 0.79), free=True)),
        _catalog(),
        _config(tmp_path, 0.002),
    )
    assert mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_JOINT, "arm/joint1") >= 0
    robot = create_robot(world, "arm")
    assert isinstance(robot.arm, Arm)
    assert isinstance(robot.gripper, Gripper)
    assert robot.base is None
    arm = robot.arm
    assert arm is not None and robot.gripper is not None
    assert arm.dof == 7
    assert arm.joint_names[0] == "arm/joint1"
    home = arm.move_joints(_HOME, speed_scale=1.0, timeout_s=4.0)
    assert home.success, home.message
    targets = (
        (0.25, -0.2, 0.0, 0.15, 0.0, 0.0, 0.0),
        (-0.3, 0.1, 0.2, -0.1, 0.2, 0.0, 0.0),
        (0.1, -0.05, -0.1, 0.05, 0.0, -0.1, 0.2),
    )
    for delta in targets:
        goal = [base + offset for base, offset in zip(_HOME, delta, strict=True)]
        result = arm.move_joints(goal, speed_scale=1.0, timeout_s=6.0)
        assert result.success, result.message
        assert result.final_rot_error_rad < 2e-3
    with pytest.raises(ConfigError, match="speed_scale"):
        arm.move_joints(_HOME, speed_scale=0.0)
    with pytest.raises(ConfigError, match="dof"):
        arm.move_joints([0.0])
    limited = arm.move_joints((0.0, 3.0, 0.0, 0.9, 0.0, 1.0, 0.0))
    assert limited.reason is FailureReason.JOINT_LIMIT
    with pytest.raises(NotImplementedError, match="Implemented in T4"):
        arm.move_pose(_pose())
    arm.stop()


@pytest.mark.assets
def test_xarm_collision_and_gripper(tmp_path: Path) -> None:
    block = ObjectSpec(
        name="block",
        collision=[PrimitiveSpec(type=PrimitiveKind.BOX, size_m=(0.12, 0.12, 0.4))],
        pose=_pose(0.35, 0.2, 1.0),
        mass_kg=1.0,
        free=False,
    )
    world = SimWorld(_xarm_scene(block), _catalog(), _config(tmp_path, 0.002))
    robot = create_robot(world, "arm")
    assert robot.arm is not None and robot.gripper is not None
    goal = [value + offset for value, offset in zip(_HOME, (0.9, 0, 0, 0, 0, 0, 0), strict=True)]
    hit = robot.arm.move_joints(goal, speed_scale=1.0, timeout_s=4.0)
    assert hit.success is False
    assert hit.reason is FailureReason.COLLISION
    opened = robot.gripper.open()
    assert opened.success
    assert robot.gripper.get_opening() == pytest.approx(1.0, abs=1e-3)
    assert robot.gripper.is_grasping() is False
    closed = robot.gripper.close(force_scale=0.4)
    assert closed.success
    assert robot.gripper.get_opening() == pytest.approx(0.0, abs=1e-3)
    rushed = robot.arm.move_joints(_HOME, speed_scale=1.0, timeout_s=0.0)
    assert rushed.reason is FailureReason.TIMEOUT


def test_diffdrive_move_to(tmp_path: Path) -> None:
    world = SimWorld(
        SceneSpec(
            floor_size_m=(12.0, 12.0),
            robots=[
                RobotPlacement(
                    robot_id="generic_diffdrive",
                    instance="base",
                    base_pose=_pose(),
                )
            ],
        ),
        _catalog(),
        _config(tmp_path, 0.01),
    )
    robot = create_robot(world, "base")
    assert isinstance(robot.base, MobileBase)
    assert robot.base is not None
    result = robot.base.move_to(Pose2D(x_m=1.0, y_m=0.5, yaw_rad=1.57), timeout_s=30.0)
    assert result.success, result.message
    pose = robot.base.get_pose2d()
    assert math.hypot(pose.x_m - 1.0, pose.y_m - 0.5) <= 0.02
    assert abs((pose.yaw_rad - 1.57 + math.pi) % (2 * math.pi) - math.pi) <= 0.05
    assert result.final_pos_error_m <= 0.02
    assert result.final_rot_error_rad <= 0.05
    robot.base.stop()
    timed = robot.base.move_to(Pose2D(x_m=3.0, y_m=0.0, yaw_rad=0.0), timeout_s=0.0)
    assert timed.reason is FailureReason.TIMEOUT
    with pytest.raises(ConfigError, match="not in the scene"):
        create_robot(world, "missing")


@pytest.mark.render
def test_camera_shapes_and_recorder(tmp_path: Path) -> None:
    camera = CameraSpec(name="overhead", pose=_pose(z=1.2), fovy_deg=50.0, width=64, height=48)
    world = SimWorld(
        SceneSpec(cameras=[camera], objects=[_cube("cube", _pose(z=0.05))]),
        _catalog(),
        _config(tmp_path, 0.01),
    )
    device = create_camera(world, "overhead")
    assert isinstance(device, Camera)
    frame = device.capture()
    assert frame.rgb.shape == (48, 64, 3)
    assert frame.rgb.dtype == np.uint8
    assert frame.depth_m is not None
    assert frame.depth_m.shape == (48, 64)
    assert frame.depth_m.dtype == np.float32
    assert frame.intrinsics.width == 64
    assert frame.intrinsics.fx == pytest.approx(frame.intrinsics.fy)
    expected_fy = 0.5 * 48 / math.tan(math.radians(50.0) / 2.0)
    assert frame.intrinsics.fy == pytest.approx(expected_fy)
    assert float(frame.depth_m[24, 32]) > 0.5
    with pytest.raises(ConfigError):
        create_camera(world, "nope")
    with pytest.raises(ConfigError):
        Recorder(world, "overhead", fps=0.0)
    with Recorder(world, "overhead", fps=10.0, path=tmp_path / "overhead.mp4") as recorder:
        world.step(5)
        recorder.capture()
        world.step(5)
        recorder.capture()
    assert recorder.path.is_file()
    assert recorder.path.stat().st_size > 0
