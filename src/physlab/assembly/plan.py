"""Assembly plans and structural checks.

``Step`` is a discriminated union on ``op``. ``validate_plan`` rejects unknown
components, a second placement of the same part, and place targets the xArm
cannot reach.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from physlab.core.errors import ConfigError
from physlab.core.types import Pose
from physlab.fabrication.cad import PART_LIBRARY
from physlab.fabrication.inventory import Inventory
from physlab.fabrication.models import InventoryItem
from physlab.sim.world import SimWorld

_ANCHORS = {"baseplate", "table"}


class ComponentRef(BaseModel):
    """One part the plan may pick up.

    Attributes:
        id: Name used by the steps.
        part_name: Library part.
        aruco_id: Marker id printed on that copy.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    id: str
    part_name: str
    aruco_id: int = Field(ge=0)


class PickStep(BaseModel):
    """Pick up ``component``."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    op: Literal["pick"]
    component: str


class PlaceStep(BaseModel):
    """Place ``component`` at ``target``."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    op: Literal["place"]
    component: str
    target: Pose
    tol_pos_m: float = Field(default=0.001, gt=0)
    tol_rot_rad: float = Field(default=0.02, gt=0)


class FastenStep(BaseModel):
    """Weld ``component`` to ``to`` (a magnetic seat or a snap)."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    op: Literal["fasten"]
    component: str
    to: str


class AlignStep(BaseModel):
    """Drive adjustment screws to maximize ``objective``."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    op: Literal["align"]
    component: str
    dof: list[str]
    objective: str
    max_iters: int = Field(gt=0)
    method: Literal["coordinate_search", "nelder_mead"] = "coordinate_search"


class InspectStep(BaseModel):
    """Check an image. ``camera`` may be ``microscope``."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    op: Literal["inspect"]
    camera: str
    check: str


Step = Annotated[
    PickStep | PlaceStep | FastenStep | AlignStep | InspectStep,
    Field(discriminator="op"),
]


class AssemblyPlan(BaseModel):
    """An ordered list of assembly steps."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    id: str
    components: list[ComponentRef]
    steps: list[Step]


def load_plan(path: Path) -> AssemblyPlan:
    """Read an assembly plan from YAML.

    Args:
        path: Plan file.

    Returns:
        Validated plan.

    Raises:
        ConfigError: The file is missing or does not match the schema.
    """
    if not path.is_file():
        raise ConfigError("Assembly plan was not found", details={"path": str(path)})
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError("Could not read the assembly plan", details={"path": str(path)}) from exc
    except yaml.YAMLError as exc:
        raise ConfigError("Assembly plan YAML is invalid", details={"path": str(path)}) from exc
    try:
        return AssemblyPlan.model_validate(loaded)
    except ValidationError as exc:
        raise ConfigError(
            "Assembly plan is invalid",
            details={"path": str(path), "errors": str(exc)},
        ) from exc


def validate_plan(plan: AssemblyPlan, inventory: Inventory, catalog: object) -> list[str]:
    """Return every problem in ``plan``.

    An empty list means the plan can be attempted. Reachability uses inverse
    kinematics on ``ufactory_xarm7`` when that model is on disk.

    Args:
        plan: Plan to check.
        inventory: Printed copies. A component is known when some item has its
            ``part_name`` and ``aruco_id``.
        catalog: Robot catalog used for the reachability check.

    Returns:
        Human-readable issues. Order is components, then steps, then targets.
    """
    issues = _unknown_components(plan, inventory)
    issues.extend(_duplicate_placements(plan))
    issues.extend(_unreachable(plan, catalog))
    return issues


def _unknown_components(plan: AssemblyPlan, inventory: Inventory) -> list[str]:
    known = {item.id for item in plan.components}
    issues: list[str] = []
    for component in plan.components:
        if component.part_name not in PART_LIBRARY:
            issues.append(f"unknown component {component.id}")
            continue
        if _match_inventory(inventory, component) is None:
            issues.append(f"unknown component {component.id}")
    for step in plan.steps:
        name = _step_component(step)
        if name is not None and name not in known:
            issues.append(f"unknown component {name}")
        target = getattr(step, "to", None)
        if isinstance(target, str) and not _known_anchor(target, known):
            issues.append(f"unknown component {target}")
    return issues


def _step_component(step: Step) -> str | None:
    if isinstance(step, InspectStep):
        return None
    return step.component


def _known_anchor(name: str, components: set[str]) -> bool:
    return name in components or name in _ANCHORS or name.endswith("_carriage")


def _match_inventory(inventory: Inventory, component: ComponentRef) -> InventoryItem | None:
    direct = inventory.get(component.id)
    if direct is not None and direct.part_name == component.part_name:
        return direct
    for item in inventory.items():
        if item.part_name == component.part_name and item.aruco_id == component.aruco_id:
            return item
    return None


def _duplicate_placements(plan: AssemblyPlan) -> list[str]:
    issues: list[str] = []
    placed: list[tuple[str, Pose]] = []
    for step in plan.steps:
        if not isinstance(step, PlaceStep):
            continue
        if any(name == step.component for name, _pose in placed):
            issues.append(f"duplicate placement of {step.component}")
        for name, pose in placed:
            if math.dist(pose.position_m, step.target.position_m) < 0.02:
                issues.append(f"duplicate placement of {step.component} and {name}")
        placed.append((step.component, step.target))
    return issues


def _unreachable(plan: AssemblyPlan, catalog: object) -> list[str]:
    places = [step for step in plan.steps if isinstance(step, PlaceStep)]
    if not places:
        return []
    world = _arm_world(catalog)
    if world is None:
        return []
    from physlab.motion.ik import solve_ik

    issues: list[str] = []
    for step in places:
        target = step.target.model_copy(
            update={
                "position_m": (
                    step.target.position_m[0],
                    step.target.position_m[1],
                    step.target.position_m[2] + 0.02,
                ),
                "quat_wxyz": (0.0, 1.0, 0.0, 0.0),
            }
        )
        solved = solve_ik(world, "arm", target, max_iters=40, pos_tol_m=0.01, rot_tol_rad=0.2)
        if not solved.success:
            issues.append(f"unreachable target for {step.component}")
    return issues


def _arm_world(catalog: object) -> SimWorld | None:
    from physlab.core.config import load_config
    from physlab.robots.catalog import RobotCatalog
    from physlab.sim.scene import RobotPlacement, SceneSpec, TableSpec
    from physlab.sim.world import SimWorld

    if not isinstance(catalog, RobotCatalog):
        return None
    try:
        spec = catalog.get("ufactory_xarm7")
    except ConfigError:
        return None
    if not spec.available:
        return None
    loaded = load_config()
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
    return SimWorld(scene, catalog, loaded.model_copy(update={"seed": 0}))
