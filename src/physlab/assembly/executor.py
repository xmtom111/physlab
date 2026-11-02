"""Run an assembly plan and write every step to the event log."""

from __future__ import annotations

import time
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from physlab.assembly.fine_align import FineAlignTool
from physlab.assembly.plan import (
    AlignStep,
    AssemblyPlan,
    FastenStep,
    InspectStep,
    PickStep,
    PlaceStep,
    validate_plan,
)
from physlab.assembly.skills import GraspMemory, PartHandle, marker_in_body, pick, place
from physlab.core.errors import ConfigError
from physlab.core.events import Actor, EventLog
from physlab.core.types import MotionResult
from physlab.fabrication.inventory import Inventory
from physlab.fabrication.models import PartArtifact
from physlab.hal.interfaces import Robot
from physlab.optics.microscope import MicroscopeModel
from physlab.perception.focus import variance_of_laplacian
from physlab.perception.localize import Localizer
from physlab.robots.catalog import RobotCatalog
from physlab.sim.world import SimWorld

_FOCUS_M = 20e-6


class StepRecord(BaseModel):
    """One executed step."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    index: int
    op: str
    success: bool
    message: str


class AssemblyReport(BaseModel):
    """Outcome of :meth:`AssemblyExecutor.run`."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    success: bool
    total_time_s: float
    steps: list[StepRecord] = Field(default_factory=list)
    failed_step: int | None = None


class AssemblyExecutor:
    """Execute a plan on one robot, stopping at the first failed step."""

    def __init__(
        self,
        robot: Robot,
        localizer: Localizer,
        inventory: Inventory,
        event_log: EventLog,
        world: SimWorld,
        catalog: RobotCatalog,
        artifacts: dict[str, PartArtifact],
        *,
        align_tool: FineAlignTool | None = None,
        microscope: MicroscopeModel | None = None,
    ) -> None:
        """Store the devices a plan can use.

        Args:
            robot: Arm, gripper, and wrist camera.
            localizer: Overhead localization.
            inventory: Printed copies.
            event_log: Log that receives one event per step.
            world: Simulation used for welds and body poses.
            catalog: Passed through to :func:`validate_plan`.
            artifacts: Part geometry keyed by component id.
            align_tool: Screw tool for ``align`` steps.
            microscope: Optical model for ``align`` and ``inspect``.
        """
        self.robot = robot
        self.localizer = localizer
        self.inventory = inventory
        self.event_log = event_log
        self.world = world
        self.catalog = catalog
        self.artifacts = artifacts
        self.align_tool = align_tool
        self.microscope = microscope
        self._memory = GraspMemory()

    def run(self, plan: AssemblyPlan) -> AssemblyReport:
        """Validate ``plan`` and execute it.

        Args:
            plan: Steps to run.

        Returns:
            Report. ``failed_step`` is the index of the first failure.
        """
        started = time.perf_counter()
        issues = validate_plan(plan, self.inventory, self.catalog)
        if issues:
            self._log("validate", -1, False, "; ".join(issues))
            return AssemblyReport(
                success=False,
                total_time_s=time.perf_counter() - started,
                failed_step=-1,
                steps=[
                    StepRecord(index=-1, op="validate", success=False, message="; ".join(issues))
                ],
            )
        records: list[StepRecord] = []
        for index, step in enumerate(plan.steps):
            ok, message = self._step(plan, step)
            records.append(StepRecord(index=index, op=step.op, success=ok, message=message))
            self._log(step.op, index, ok, message, component=getattr(step, "component", ""))
            if not ok:
                return AssemblyReport(
                    success=False,
                    total_time_s=time.perf_counter() - started,
                    steps=records,
                    failed_step=index,
                )
        return AssemblyReport(
            success=True, total_time_s=time.perf_counter() - started, steps=records
        )

    def _step(self, plan: AssemblyPlan, step: object) -> tuple[bool, str]:
        if isinstance(step, PickStep):
            return _as_bool(
                pick(self.robot, self.localizer, self._handle(plan, step.component), self._memory)
            )
        if isinstance(step, PlaceStep):
            return _as_bool(
                place(
                    self.robot,
                    self.localizer,
                    self._handle(plan, step.component),
                    step.target,
                    step.tol_pos_m,
                    step.tol_rot_rad,
                    self._memory,
                )
            )
        if isinstance(step, FastenStep):
            return self._fasten(plan, step)
        if isinstance(step, AlignStep):
            return self._align(step)
        if isinstance(step, InspectStep):
            return self._inspect(step)
        return False, "unknown step"

    def _handle(self, plan: AssemblyPlan, component_id: str) -> PartHandle:
        component = next(item for item in plan.components if item.id == component_id)
        artifact = self.artifacts[component_id]
        item = next(
            entry
            for entry in self.inventory.items()
            if entry.part_name == component.part_name and entry.aruco_id == component.aruco_id
        )
        ee_body = self.robot.spec.sim.ee_body or "link7"
        prefix = self.robot.arm.name if self.robot.arm is not None else "arm"
        return PartHandle(
            id=component.id,
            aruco_id=component.aruco_id,
            body=item.part_instance_id,
            grasp=artifact.frames["grasp"],
            marker=marker_in_body(artifact),
            world=self.world,
            ee_body=f"{prefix}/{ee_body}",
        )

    def _fasten(self, plan: AssemblyPlan, step: FastenStep) -> tuple[bool, str]:
        handle = self._handle(plan, step.component)
        if step.to in {"baseplate", "table"} or step.to.endswith("_carriage"):
            target = step.to
        else:
            target = self._handle(plan, step.to).body
        if target.endswith("_carriage"):
            _zero_screw(self.world, target.removesuffix("_carriage"))
        try:
            self.world.weld(handle.body, target)
        except ConfigError as exc:
            return False, exc.message
        return True, f"fastened to {target}"

    def _align(self, step: AlignStep) -> tuple[bool, str]:
        if self.align_tool is None or self.microscope is None:
            return False, "alignment needs a tool and a microscope"
        docked = self.align_tool.dock(self.robot, step.component)
        if not docked.success:
            return False, docked.message
        score = self.align_tool.align(
            lambda: (
                variance_of_laplacian(self.microscope.render().image) if self.microscope else 0.0
            ),
            step.dof,
            method=step.method,
            max_iters=step.max_iters,
        )
        self.align_tool.undock()
        if step.objective != "variance_of_laplacian":
            return False, f"unknown objective {step.objective}"
        return True, f"score {score:.4f}"

    def _inspect(self, step: InspectStep) -> tuple[bool, str]:
        if step.camera != "microscope" or self.microscope is None:
            return False, "inspect needs the microscope model"
        view = self.microscope.render()
        if step.check == "focused":
            ok = abs(view.defocus_m) < _FOCUS_M and view.illumination > 0.2
            return ok, f"defocus {view.defocus_m:.6f} illumination {view.illumination:.2f}"
        return False, f"unknown check {step.check}"

    def _log(self, op: str, index: int, ok: bool, message: str, component: str = "") -> None:
        self.event_log.record(
            Actor.AGENT,
            f"assembly_{op}",
            params={"index": index, "component": component},
            result={"success": ok, "message": message},
        )


def _zero_screw(world: SimWorld, stage: str) -> None:
    """Put a focus stage back at mid-travel before the sample is welded to it."""
    joint = f"{stage}_z"
    world.set_joint_position(joint, 0.0)
    world.set_actuator_ctrl(joint, 0.0)


def _as_bool(result: MotionResult) -> tuple[bool, str]:
    return result.success, result.message or ("ok" if result.success else "failed")
