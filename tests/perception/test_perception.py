"""ArUco localization and focus metrics."""

from __future__ import annotations

import cv2
import mujoco
import numpy as np
import pytest

from physlab.core.config import load_config
from physlab.core.frames import rpy_to_quat
from physlab.core.types import Pose
from physlab.hal.factory import create_camera
from physlab.perception.aruco import detect, marker_png
from physlab.perception.focus import centroid_of_bright_spot, tenengrad, variance_of_laplacian
from physlab.perception.localize import Localizer
from physlab.robots.catalog import RobotCatalog
from physlab.sim.scene import (
    CameraSpec,
    ObjectSpec,
    PrimitiveKind,
    PrimitiveSpec,
    SceneSpec,
    TableSpec,
    add_overhead_stereo,
    add_wrist_camera,
)
from physlab.sim.world import SimWorld


def _world() -> tuple[SimWorld, Localizer]:
    loaded = load_config()
    marker = ObjectSpec(
        name="tag",
        collision=[PrimitiveSpec(type=PrimitiveKind.BOX, size_m=(0.08, 0.08, 0.02))],
        pose=Pose(position_m=(0.0, 0.0, 0.76), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        mass_kg=0.1,
        rgba=(0.15, 0.15, 0.15, 1.0),
        free=True,
        aruco_id=12,
        aruco_size_m=0.05,
    )
    scene = SceneSpec(table=TableSpec(), objects=[marker])
    add_overhead_stereo(scene, height_m=0.98, baseline_m=0.06, width=1280, height=960)
    scene.cameras.append(
        CameraSpec(
            name="wrist",
            pose=Pose(position_m=(0.0, 0.0, 0.9), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
            fovy_deg=40.0,
            width=1280,
            height=960,
        )
    )
    world = SimWorld(scene, RobotCatalog(loaded.configs_dir, loaded.assets_dir), loaded)
    cameras = [create_camera(world, name) for name in ("overhead_left", "overhead_right")]
    return world, Localizer(cameras, 0.05)


def _place(world: SimWorld, pose: Pose) -> None:
    joint = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_JOINT, "tag_free")
    address = int(world.model.jnt_qposadr[joint])
    world.data.qpos[address : address + 3] = pose.position_m
    world.data.qpos[address + 3 : address + 7] = pose.quat_wxyz
    world.data.qvel[:] = 0.0
    mujoco.mj_forward(world.model, world.data)


def _geom_pose(world: SimWorld) -> Pose:
    geom = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_GEOM, "tag_aruco")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(world.data.geom_xmat[geom], dtype=np.float64).reshape(3, 3)
    matrix[:3, 3] = np.asarray(world.data.geom_xpos[geom], dtype=np.float64)
    return Pose.from_matrix(matrix)


@pytest.mark.render
def test_stereo_localization_on_seeded_poses() -> None:
    world, localizer = _world()
    rng = np.random.default_rng(0)
    for _ in range(10):
        yaw = float(rng.uniform(-np.pi, np.pi))
        pose = Pose(
            position_m=(
                float(rng.uniform(-0.03, 0.03)),
                float(rng.uniform(-0.03, 0.03)),
                0.76,
            ),
            quat_wxyz=tuple(float(v) for v in rpy_to_quat(0.0, 0.0, yaw)),
        )
        _place(world, pose)
        truth = _geom_pose(world)
        found = localizer.locate(12)
        assert found is not None
        assert found.n_views == 2
        error = found.pose_world.error_to(truth)
        assert error[0] < 0.002
        assert error[1] < np.deg2rad(1.0)


@pytest.mark.render
def test_wrist_refine_improves_a_coarse_pose() -> None:
    world, localizer = _world()
    _place(world, Pose(position_m=(0.01, -0.01, 0.76), quat_wxyz=(1.0, 0.0, 0.0, 0.0)))
    truth = _geom_pose(world)
    coarse = Pose(
        position_m=(truth.position_m[0] + 0.008, truth.position_m[1] - 0.006, truth.position_m[2]),
        quat_wxyz=truth.quat_wxyz,
    )
    wrist = create_camera(world, "wrist")
    refined = localizer.refine_with_wrist(wrist, 12, coarse)
    coarse_error = coarse.error_to(truth)[0]
    refined_error = refined.error_to(truth)[0]
    assert refined_error < coarse_error
    assert refined_error < 0.002


def test_focus_falls_as_blur_increases() -> None:
    image = np.zeros((160, 160), dtype=np.uint8)
    image[40:120, 40:120] = 255
    image[70:90, 50:110] = 0
    image[50:110, 70:90] = 0
    blurred = [image]
    blurred.extend(cv2.GaussianBlur(image, (0, 0), sigma) for sigma in (1.0, 2.0, 4.0))
    for metric in (variance_of_laplacian, tenengrad):
        scores = [metric(frame) for frame in blurred]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] > scores[-1]


def test_centroid_of_a_bright_blob() -> None:
    image = np.zeros((80, 100), dtype=np.uint8)
    image[30:50, 60:80] = 255
    center = centroid_of_bright_spot(image)
    assert center is not None
    assert center[0] == pytest.approx(69.5, abs=1.0)
    assert center[1] == pytest.approx(39.5, abs=1.0)


def test_wrist_camera_helper_names_the_end_effector() -> None:
    scene = SceneSpec()
    add_wrist_camera(scene, "arm")
    camera = scene.cameras[-1]
    assert camera.name == "arm_wrist"
    assert camera.attached_to_body == "arm/link7"
    assert camera.pose is not None
    assert camera.pose.quat_wxyz == (0.0, 1.0, 0.0, 0.0)


def test_marker_png_is_cached() -> None:
    from physlab.hal.interfaces import CameraIntrinsics, Frame

    first = marker_png(3, 64)
    second = marker_png(3, 64)
    assert first == second
    assert first.is_file()
    gray = cv2.imread(str(first), cv2.IMREAD_GRAYSCALE)
    padded = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    rgb = cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)
    height, width = rgb.shape[:2]
    frame = Frame(
        rgb=rgb,
        depth_m=None,
        intrinsics=CameraIntrinsics(
            fx=1, fy=1, cx=width / 2, cy=height / 2, width=width, height=height
        ),
        pose_in_world=None,
        timestamp_s=0.0,
    )
    assert [item.id for item in detect(frame)] == [3]
