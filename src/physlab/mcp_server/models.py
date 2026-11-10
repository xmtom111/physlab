"""Pydantic inputs for the MCP tools.

The server publishes each model as the ``request`` property of the tool
schema. Lengths are meters, angles are radians, and quaternions are
``(w, x, y, z)`` unless a field name says otherwise. CAD parameters stay in
millimeters because that is the fabrication schema.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Input(BaseModel):
    """Shared config: unknown fields are rejected."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class ResetSceneInput(_Input):
    """Seed for :func:`SimWorld.reset`.

    Attributes:
        seed: Non-negative seed. The session seed is used when omitted.
        approval_token: Required in real mode.
    """

    seed: int | None = Field(default=None, ge=0)
    approval_token: str | None = None


class LoadSceneInput(_Input):
    """Scene stem under ``configs/scenes``.

    Attributes:
        name: File stem, for example ``table_gantry``.
    """

    name: str = Field(min_length=1)


class RobotInstanceInput(_Input):
    """One robot instance in the loaded scene.

    Attributes:
        instance: ``RobotPlacement.instance``, for example ``arm`` or ``gantry``.
    """

    instance: str = Field(min_length=1)


class MoveArmJointsInput(_Input):
    """Joint-space target.

    Attributes:
        instance: Robot instance.
        q_rad: Joint targets in catalog order. Revolute joints are radians.
            Prismatic gantry joints are meters.
        speed_scale: Fraction of the velocity limit, in ``(0, 1]``.
        timeout_s: Wall-clock budget, seconds.
        approval_token: Required in real mode.
    """

    instance: str = Field(min_length=1)
    q_rad: list[float] = Field(min_length=1)
    speed_scale: float = Field(default=0.5, gt=0, le=1)
    timeout_s: float = Field(default=10.0, gt=0)
    approval_token: str | None = None


class MoveArmPoseInput(_Input):
    """End-effector pose in a named frame.

    Attributes:
        instance: Robot instance.
        position_m: Origin in ``frame``, meters.
        quat_wxyz: Unit quaternion ``(w, x, y, z)``.
        frame: Reference frame. ``world`` is the scene frame.
        speed_scale: Fraction of the velocity limit, in ``(0, 1]``.
        timeout_s: Wall-clock budget, seconds.
        approval_token: Required in real mode.
    """

    instance: str = Field(min_length=1)
    position_m: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    frame: str = "world"
    speed_scale: float = Field(default=0.5, gt=0, le=1)
    timeout_s: float = Field(default=10.0, gt=0)
    approval_token: str | None = None


class GripperInput(_Input):
    """Gripper command.

    Attributes:
        instance: Robot instance.
        action: ``open`` or ``close``.
        force_scale: Close effort scale in ``(0, 1]``. Ignored by the sim gripper.
        approval_token: Required in real mode.
    """

    instance: str = Field(min_length=1)
    action: Literal["open", "close"]
    force_scale: float = Field(default=0.5, gt=0, le=1)
    approval_token: str | None = None


class MoveBaseInput(_Input):
    """Planar base target in the world frame.

    Attributes:
        instance: Robot instance.
        x_m: World X, meters.
        y_m: World Y, meters.
        yaw_rad: Heading about world Z, radians.
        approval_token: Required in real mode.
    """

    instance: str = Field(min_length=1)
    x_m: float
    y_m: float
    yaw_rad: float
    approval_token: str | None = None


class StopAllInput(_Input):
    """Stop request. A token is not required, including in real mode.

    Attributes:
        approval_token: Ignored. Stopping does not need approval.
    """

    approval_token: str | None = None


class CaptureImageInput(_Input):
    """Camera name in the loaded scene.

    Attributes:
        camera: ``CameraSpec.name``, for example ``overhead_left``.
    """

    camera: str = Field(min_length=1)


class LocateMarkerInput(_Input):
    """ArUco id to fuse across the scene cameras.

    Attributes:
        marker_id: Marker id, matching ``DICT_4X4_50``.
        marker_size_m: Outer marker side, including the black border, meters.
    """

    marker_id: int = Field(ge=0)
    marker_size_m: float = Field(default=0.03, gt=0)


class DesignPartInput(_Input):
    """Parametric part. Parameter units are millimeters except where noted.

    Attributes:
        part_name: Library name from ``list_parts``.
        params: Overrides. Omitted keys use the part schema defaults.
    """

    part_name: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class CheckPartInput(_Input):
    """A design produced by ``design_part``.

    Attributes:
        design_id: Id returned by ``design_part``.
    """

    design_id: str = Field(min_length=1)


class PrintPartInput(_Input):
    """Mock print of a design. The sim service is accelerated.

    Attributes:
        design_id: Id returned by ``design_part``.
        material: Filament name. The mock service records it and does not
            change the mesh.
        quality: Profile name. Recorded only.
        copies: How many copies to print.
        approval_token: Required in real mode.
    """

    design_id: str = Field(min_length=1)
    material: str = "PLA"
    quality: str = "standard"
    copies: int = Field(default=1, ge=1)
    approval_token: str | None = None


class AssemblyPlanInput(_Input):
    """Assembly plan JSON. See ``examples/plans/microscope_v0.yaml``.

    Attributes:
        plan: Object with ``id``, ``components``, and ``steps``.
            Place targets are meters in the world frame.
        approval_token: Required in real mode for ``run_assembly_plan``.
            Ignored by ``validate_assembly_plan``.
    """

    plan: dict[str, Any]
    approval_token: str | None = None


class ScenarioInput(_Input):
    """Controller scenario name.

    Attributes:
        scenario: ``focus_stage``, ``pendulum_release``, or ``line_follow``.
    """

    scenario: str = Field(min_length=1)


class CheckCodeInput(_Input):
    """Controller source for the static allowlist.

    Attributes:
        source: Python source that subclasses ``Controller``.
    """

    source: str


class RunControllerInput(_Input):
    """Controller run. Time is seconds. Channel units come from the manifest.

    Attributes:
        source: Python source.
        scenario: Scenario name from ``get_device_manifest``.
        duration_s: Simulated time, seconds.
        params: Numeric parameters passed to the controller context.
        seed: Sensor and scenario seed.
        approval_token: Required in real mode.
    """

    source: str
    scenario: str = Field(min_length=1)
    duration_s: float = Field(gt=0)
    params: dict[str, float] = Field(default_factory=dict)
    seed: int = Field(default=0, ge=0)
    approval_token: str | None = None


class ProtocolInput(_Input):
    """Experiment or chemistry protocol JSON.

    Attributes:
        protocol: Protocol object. Experiment protocols use SI units named on
            each variable. Chemistry concentrations are molar and volumes are
            microliters.
        approval_token: Required in real mode for a run. Ignored by the
            validate and check tools.
    """

    protocol: dict[str, Any]
    approval_token: str | None = None


class RunExperimentInput(ProtocolInput):
    """Experiment protocol plus the approval token used in real mode."""


class RunChemInput(ProtocolInput):
    """Chemistry protocol and the question stored on the run.

    Attributes:
        question: Question copied onto the run result.
        hypothesis: Hypothesis copied onto the run result.
        seed: Plate-reader seed.
        analyte: Species recorded as ``concentration_m``.
    """

    question: str = "What does this protocol measure?"
    hypothesis: str = "The virtual lab follows the allowlisted chemistry."
    seed: int = Field(default=0, ge=0)
    analyte: str = "food_dye"


class FitModelsInput(_Input):
    """Columns to fit on a finished run.

    Attributes:
        run_id: Run id returned by ``run_experiment`` or ``run_chem_protocol``.
        x: Independent column name.
        y: Dependent column name.
    """

    run_id: str = Field(min_length=1)
    x: str = Field(min_length=1)
    y: str = Field(min_length=1)


class GetReportInput(_Input):
    """Report request. ``x`` and ``y`` are inferred when a trial has one of each.

    Attributes:
        run_id: Run id.
        x: Independent column. Inferred when omitted.
        y: Dependent column. Inferred when omitted.
    """

    run_id: str = Field(min_length=1)
    x: str | None = None
    y: str | None = None


class GetJobInput(_Input):
    """Job id returned by a long-running tool.

    Attributes:
        job_id: Id such as ``job-1``.
    """

    job_id: str = Field(min_length=1)
