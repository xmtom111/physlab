"""Build, print, and assemble the tabletop microscope."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from physlab.assembly.executor import AssemblyExecutor, AssemblyReport
from physlab.assembly.fine_align import AdjustableDof, FineAlignTool
from physlab.assembly.plan import AssemblyPlan, PlaceStep, load_plan
from physlab.core.config import PhysLabConfig, load_config
from physlab.core.events import EventLog
from physlab.core.logging import get_logger
from physlab.core.types import Pose
from physlab.fabrication.inventory import Inventory
from physlab.fabrication.mesh_check import load_printer_profile
from physlab.fabrication.mock_print import MockPrintService
from physlab.fabrication.models import PartArtifact
from physlab.fabrication.print_service import fabricate
from physlab.hal.factory import create_robot
from physlab.optics.microscope import MicroscopeModel
from physlab.perception.localize import Localizer
from physlab.robots.catalog import RobotCatalog
from physlab.sim.recorder import Recorder
from physlab.sim.scene import (
    CameraSpec,
    FocusStageSpec,
    RobotPlacement,
    SceneSpec,
    TableSpec,
    add_wrist_camera,
)
from physlab.sim.world import SimWorld

_LOG = get_logger(__name__)
_IDENTITY = (1.0, 0.0, 0.0, 0.0)
_STAGE = "sample_stage_focus"


def microscope_plan() -> Path:
    """Return the path of the checked-in microscope plan.

    Returns:
        ``examples/plans/microscope_v0.yaml``.
    """
    return Path(__file__).resolve().parents[3] / "examples" / "plans" / "microscope_v0.yaml"


def assemble_microscope(
    seed: int,
    *,
    plan_path: Path | None = None,
    save_dir: Path | None = None,
) -> AssemblyReport:
    """Mock-print the optical parts, assemble them, and focus the sample.

    Args:
        seed: Shifts the pickup slots and the IK restarts.
        plan_path: Plan YAML. Defaults to :func:`microscope_plan`.
        save_dir: When set, write ``microscope.png`` and ``assembly.mp4``.

    Returns:
        The executor report.
    """
    loaded = load_config().model_copy(update={"seed": seed})
    loaded = loaded.model_copy(update={"sim": loaded.sim.model_copy(update={"timestep_s": 0.002})})
    plan = load_plan(plan_path or microscope_plan())
    world, catalog = _world(loaded, _sample_target(plan))
    robot = create_robot(world, "arm")
    localizer = Localizer(
        [robot.cameras[name] for name in ("overhead_left", "overhead_right")],
        0.03,
    )
    inventory, artifacts = _print_parts(world, loaded, plan, seed)
    microscope = _microscope(world, inventory, plan)
    tool = FineAlignTool(world, f"{_STAGE}_knob", _screws())
    report = AssemblyExecutor(
        robot,
        localizer,
        inventory,
        EventLog(loaded.run_dir, seed=seed),
        world,
        catalog,
        artifacts,
        align_tool=tool,
        microscope=microscope,
    ).run(plan)
    if save_dir is not None and report.success:
        _save(world, microscope, save_dir)
    _LOG.info("assembly_done", seed=seed, success=report.success, failed_step=report.failed_step)
    return report


def _sample_target(plan: AssemblyPlan) -> Pose:
    for step in plan.steps:
        if isinstance(step, PlaceStep) and step.component == "sample":
            return step.target
    return Pose(position_m=(0.40, 0.0, 0.75), quat_wxyz=_IDENTITY)


def _world(config: PhysLabConfig, sample_pose: Pose) -> tuple[SimWorld, RobotCatalog]:
    scene = SceneSpec(
        table=TableSpec(),
        robots=[
            RobotPlacement(
                robot_id="ufactory_xarm7",
                instance="arm",
                base_pose=Pose(position_m=(0.0, 0.0, 0.75), quat_wxyz=_IDENTITY),
            )
        ],
        focus_stages=[
            FocusStageSpec(
                name=_STAGE,
                pose=sample_pose,
                travel_m=0.008,
                knob_offset_m=(0.0, 0.05, 0.06),
                baseplate_pose=Pose(position_m=(0.45, 0.0, 0.742), quat_wxyz=_IDENTITY),
            )
        ],
    )
    _overhead(scene)
    add_wrist_camera(scene, "arm", ee_body="link7")
    catalog = RobotCatalog(config.configs_dir, config.assets_dir)
    return SimWorld(scene, catalog, config), catalog


def _print_parts(
    world: SimWorld,
    config: PhysLabConfig,
    plan: AssemblyPlan,
    seed: int,
) -> tuple[Inventory, dict[str, PartArtifact]]:
    rng = np.random.default_rng(seed)
    poses = [_jitter(rng, index, len(plan.components)) for index in range(len(plan.components))]
    service = MockPrintService(world, mm3_per_s=1.0e9, pickup_poses=poses)
    profile = load_printer_profile()
    inventory = Inventory()
    artifacts: dict[str, PartArtifact] = {}
    for component in plan.components:
        result = fabricate(component.part_name, {}, service, profile)
        if not result.ok or result.inventory is None or result.part is None:
            raise RuntimeError(f"mock print failed for {component.part_name}: {result.issues}")
        inventory.add(result.inventory)
        artifacts[component.id] = result.part
    del config
    return inventory, artifacts


def _jitter(rng: np.random.Generator, index: int, count: int) -> Pose:
    del count
    spots = ((0.32, -0.12), (0.44, -0.12), (0.56, -0.12), (0.36, 0.12), (0.50, 0.12))
    x_m, y_m = spots[index % len(spots)]
    return Pose(
        position_m=(
            x_m + float(rng.uniform(-0.008, 0.008)),
            y_m + float(rng.uniform(-0.008, 0.008)),
            0.75,
        ),
        quat_wxyz=_IDENTITY,
    )


def _microscope(world: SimWorld, inventory: Inventory, plan: AssemblyPlan) -> MicroscopeModel:
    names = {}
    for component in plan.components:
        item = next(
            entry
            for entry in inventory.items()
            if entry.part_name == component.part_name and entry.aruco_id == component.aruco_id
        )
        names[component.id] = item.part_instance_id
    return MicroscopeModel(
        world,
        0.02,
        0.04,
        sample_stage=names["sample"],
        objective=names["objective"],
        tube=names["tube"],
        camera_mount=names["camera"],
        led_holder=names["led"],
    )


def _overhead(scene: SceneSpec) -> None:
    for name, y_m in (("overhead_left", -0.06), ("overhead_right", 0.06)):
        scene.cameras.append(
            CameraSpec(
                name=name,
                pose=Pose(position_m=(0.45, y_m, 1.15), quat_wxyz=_IDENTITY),
                fovy_deg=60.0,
                width=960,
                height=720,
            )
        )


def _screws() -> list[AdjustableDof]:
    return [
        AdjustableDof("z", f"{_STAGE}_z", f"{_STAGE}_z", "slide"),
        AdjustableDof("rx", f"{_STAGE}_rx", f"{_STAGE}_rx", "hinge"),
    ]


def _save(world: SimWorld, microscope: MicroscopeModel, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    image = microscope.render().image
    cv2.imwrite(str(directory / "microscope.png"), image)
    with Recorder(world, "overhead_left", 10, directory / "assembly.mp4") as recorder:
        for _ in range(8):
            world.step(10)
            recorder.capture()
