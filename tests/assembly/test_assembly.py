"""Plan checks, place correction, focus, and the microscope assembly."""

from __future__ import annotations

import numpy as np
import pytest

from physlab.assembly.bench import assemble_microscope
from physlab.assembly.fine_align import AdjustableDof, FineAlignTool
from physlab.assembly.plan import (
    AssemblyPlan,
    ComponentRef,
    PickStep,
    PlaceStep,
    load_plan,
    validate_plan,
)
from physlab.assembly.skills import PartHandle, marker_in_body, place
from physlab.core.config import load_config
from physlab.core.types import Pose
from physlab.fabrication.cad import build_part
from physlab.fabrication.inventory import Inventory
from physlab.fabrication.models import InventoryItem
from physlab.hal.factory import create_robot
from physlab.optics.microscope import MicroscopeModel
from physlab.optics.paraxial import image_distance, magnification
from physlab.perception.localize import Localizer
from physlab.robots.catalog import RobotCatalog
from physlab.sim.scene import (
    CameraSpec,
    FocusStageSpec,
    ObjectSpec,
    PrimitiveKind,
    PrimitiveSpec,
    RobotPlacement,
    SceneSpec,
    TableSpec,
    add_wrist_camera,
)
from physlab.sim.world import SimWorld

_IDENTITY = (1.0, 0.0, 0.0, 0.0)


def test_validation_rejects_an_unknown_component() -> None:
    plan = AssemblyPlan(
        id="bad",
        components=[ComponentRef(id="ghost", part_name="led_holder", aruco_id=1)],
        steps=[PickStep(op="pick", component="ghost")],
    )
    issues = validate_plan(
        plan, Inventory(), RobotCatalog(load_config().configs_dir, load_config().assets_dir)
    )
    assert any("unknown component ghost" in issue for issue in issues)


@pytest.mark.assets
def test_validation_rejects_duplicate_and_unreachable_targets() -> None:
    catalog = RobotCatalog(load_config().configs_dir, load_config().assets_dir)
    inventory = Inventory()
    inventory.add(
        InventoryItem(part_instance_id="led", part_name="led_holder", aruco_id=4, location="bench")
    )
    near = _pose(0.42, 0.0, 0.75)
    plan = AssemblyPlan(
        id="twice",
        components=[ComponentRef(id="led", part_name="led_holder", aruco_id=4)],
        steps=[
            PlaceStep(op="place", component="led", target=near),
            PlaceStep(op="place", component="led", target=_pose(0.50, 0.0, 0.75)),
        ],
    )
    duplicate = validate_plan(plan, inventory, catalog)
    assert any("duplicate placement" in issue for issue in duplicate)
    far = AssemblyPlan(
        id="far",
        components=[ComponentRef(id="led", part_name="led_holder", aruco_id=4)],
        steps=[PlaceStep(op="place", component="led", target=_pose(1.6, 1.6, 0.75))],
    )
    unreachable = validate_plan(far, inventory, catalog)
    assert any("unreachable" in issue for issue in unreachable)


@pytest.mark.assets
def test_place_corrects_a_3mm_offset(tmp_path) -> None:
    artifact = build_part("lens_holder", directory=tmp_path)
    target = _pose(0.46, 0.0, 0.75)
    start = _pose(0.463, 0.0, 0.75)
    world, robot, localizer = _arm_scene(
        [
            ObjectSpec(
                name="lens",
                collision=list(artifact.collision_primitives),
                pose=start,
                mass_kg=0.05,
                free=True,
                aruco_id=12,
                aruco_size_m=0.03,
            )
        ]
    )
    handle = PartHandle(
        id="lens",
        aruco_id=12,
        body="lens",
        grasp=artifact.frames["grasp"],
        marker=marker_in_body(artifact),
        world=world,
        ee_body="arm/xarm_gripper_base_link",
    )
    result = place(robot, localizer, handle, target, tol_pos_m=0.004, tol_rot_rad=0.25)
    error = _distance(world.body_pose("lens").position_m, target.position_m)
    assert result.success, result.message
    assert error < 0.003


@pytest.mark.assets
def test_fine_align_reaches_focus_from_1mm() -> None:
    stage = _pose(0.40, 0.0, 1.098)
    boxes = [
        _fixed("objective", 2.0, 0.0, 1.098),
        _fixed("tube", 2.2, 0.0, 1.098),
        _fixed("camera_mount", 2.4, 0.0, 1.098),
        _fixed("led_holder", 2.0, 0.0, 1.098),
    ]
    world, robot, _localizer = _arm_scene(
        boxes,
        focus=FocusStageSpec(
            name="focus", pose=stage, travel_m=0.004, knob_offset_m=(0.0, 0.0, 0.0)
        ),
    )
    model = MicroscopeModel(
        world,
        0.02,
        0.04,
        sample_stage="focus_carriage",
        objective="objective",
        tube="tube",
        camera_mount="camera_mount",
        led_holder="led_holder",
    )
    world.set_joint_position("focus_z", 0.001)
    world.set_actuator_ctrl("focus_z", 0.001)
    assert abs(model.render().defocus_m - 0.001) < 5e-5
    tool = FineAlignTool(
        world,
        "focus_knob",
        [
            AdjustableDof("z", "focus_z", "focus_z", "slide"),
            AdjustableDof("rx", "focus_rx", "focus_rx", "hinge"),
        ],
    )
    assert tool.dock(robot, "sample").success
    tool.align(
        lambda: -abs(model.render().defocus_m), ["z"], method="coordinate_search", max_iters=40
    )
    assert abs(model.render().defocus_m) < 20e-6
    world.set_joint_position("focus_z", 0.001)
    world.set_actuator_ctrl("focus_z", 0.001)
    tool.docked = True
    tool.align(lambda: -abs(model.render().defocus_m), ["z"], method="nelder_mead", max_iters=40)
    assert abs(model.render().defocus_m) < 20e-6


def test_paraxial_two_focal_length_conjugate() -> None:
    image = image_distance(0.2, 0.1)
    assert image == pytest.approx(0.2)
    assert magnification(0.2, 0.1) == pytest.approx(-1.0)


@pytest.mark.slow
@pytest.mark.assets
def test_microscope_assembly_succeeds_for_at_least_four_seeds() -> None:
    successes = sum(bool(assemble_microscope(seed).success) for seed in range(5))
    assert successes >= 4


def test_checked_in_plan_loads() -> None:
    plan = load_plan(load_config().configs_dir.parent / "examples" / "plans" / "microscope_v0.yaml")
    assert plan.id == "microscope_v0"
    assert [item.part_name for item in plan.components] == [
        "led_holder",
        "sample_stage",
        "lens_holder",
        "tube",
        "camera_mount",
    ]


def _pose(x_m: float, y_m: float, z_m: float) -> Pose:
    return Pose(position_m=(x_m, y_m, z_m), quat_wxyz=_IDENTITY)


def _distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return float(np.linalg.norm(np.asarray(left) - np.asarray(right)))


def _fixed(name: str, x_m: float, y_m: float, z_m: float) -> ObjectSpec:
    return ObjectSpec(
        name=name,
        collision=[PrimitiveSpec(type=PrimitiveKind.BOX, size_m=(0.02, 0.02, 0.02))],
        pose=_pose(x_m, y_m, z_m),
        mass_kg=0.05,
    )


def _arm_scene(
    objects: list[ObjectSpec],
    focus: FocusStageSpec | None = None,
) -> tuple[SimWorld, object, Localizer]:
    loaded = load_config().model_copy(
        update={"sim": load_config().sim.model_copy(update={"timestep_s": 0.002})}
    )
    scene = SceneSpec(
        table=TableSpec(),
        robots=[
            RobotPlacement(
                robot_id="ufactory_xarm7",
                instance="arm",
                base_pose=_pose(0.0, 0.0, 0.75),
            )
        ],
        objects=objects,
        focus_stages=[] if focus is None else [focus],
    )
    for name, y_m in (("overhead_left", -0.06), ("overhead_right", 0.06)):
        scene.cameras.append(
            CameraSpec(
                name=name,
                pose=_pose(0.45, y_m, 1.15),
                fovy_deg=60.0,
                width=960,
                height=720,
            )
        )
    add_wrist_camera(scene, "arm")
    catalog = RobotCatalog(loaded.configs_dir, loaded.assets_dir)
    world = SimWorld(scene, catalog, loaded)
    robot = create_robot(world, "arm")
    localizer = Localizer(
        [robot.cameras[name] for name in ("overhead_left", "overhead_right")], 0.03
    )
    return world, robot, localizer
