"""Pydantic descriptions of robots in the physlab catalog.

Names of joints, actuators, sites, and bodies are copied from MuJoCo Menagerie
XML. Procedural robots are described the same way so the checker can confirm
the builder created those names.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RobotKind(StrEnum):
    """Role of a robot in the catalog."""

    ARM = "arm"
    MOBILE_BASE = "mobile_base"
    MOBILE_MANIPULATOR = "mobile_manipulator"
    GANTRY = "gantry"


class BaseType(StrEnum):
    """How a mobile base is commanded."""

    DIFF_DRIVE = "diff_drive"
    HOLONOMIC_JOINTS = "holonomic_joints"
    NONE = "none"


class SimSource(StrEnum):
    """Where the MuJoCo model comes from."""

    MENAGERIE = "menagerie"
    PROCEDURAL = "procedural"


class GripperSpec(BaseModel):
    """Gripper command and the geometry used for contact.

    ``attach_from`` is a MuJoCo Menagerie folder. When it is set, the gripper
    model is attached with ``MjSpec.attach`` and its element names are prefixed
    with ``"{attach_from}/"``. Names below are the names in that folder's XML,
    without the prefix.

    Attributes:
        actuator: Actuator that opens and closes the fingers.
        open_ctrl: Command that opens the gripper.
        closed_ctrl: Command that closes the gripper.
        finger_geoms: Named collision geoms on the fingers.
        attach_from: Menagerie folder to attach, or ``None`` when the gripper
            is already in the arm XML.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    actuator: str
    open_ctrl: float
    closed_ctrl: float
    finger_geoms: tuple[str, ...] = ()
    attach_from: str | None = None


class SimSpec(BaseModel):
    """Simulation description shared by menagerie and procedural robots.

    Attributes:
        source: Menagerie file or an in-code builder.
        menagerie_dir: Folder under ``assets/mujoco_menagerie``.
        xml_file: MJCF file inside that folder.
        procedural_builder: Name of the builder in ``physlab.robots.compose``.
        arm_joints: Arm or gantry joints, in kinematic order.
        arm_actuators: Actuators that drive those joints. Unnamed actuators in
            the source XML are omitted.
        ee_site: Tool site. ``None`` when the site is created at runtime.
        ee_body: Body the tool site is attached to.
        ee_offset_m: Site position in the end-effector body frame, meters.
        home_qpos_rad: Home position for ``arm_joints`` only, radians or meters
            for prismatic joints.
        gripper: Gripper command, if this robot has one.
        base_type: Mobile-base command style.
        base_joints: Driven base joints.
        base_actuators: Actuators for ``base_joints``.
        mount_site: Site where an arm can be attached.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    source: SimSource
    menagerie_dir: str | None = None
    xml_file: str | None = None
    procedural_builder: str | None = None
    arm_joints: tuple[str, ...] = ()
    arm_actuators: tuple[str, ...] = ()
    ee_site: str | None = None
    ee_body: str | None = None
    ee_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    home_qpos_rad: tuple[float, ...] = ()
    gripper: GripperSpec | None = None
    base_type: BaseType = BaseType.NONE
    base_joints: tuple[str, ...] = ()
    base_actuators: tuple[str, ...] = ()
    mount_site: str | None = None


class RealSpec(BaseModel):
    """Optional real-hardware driver. Drivers themselves arrive in later tasks.

    Attributes:
        driver: Registry name used by the real backend, or ``None``.
        sdk_package: Python distribution that provides the SDK, or ``None``.
        notes: Extra hardware notes. Empty when unused.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    driver: str | None = None
    sdk_package: str | None = None
    notes: str = ""


class RobotSpec(BaseModel):
    """One catalog entry.

    Attributes:
        id: Stable id, also the YAML file stem.
        display_name: Human-readable name.
        kind: Arm, base, mobile manipulator, or gantry.
        license: License of the upstream model, or MIT for procedural robots.
        sim: Simulation description.
        real: Real-hardware description.
        available: False when the menagerie folder is not in the pinned commit.
        unavailable_reason: Why ``available`` is false.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    id: str
    display_name: str
    kind: RobotKind
    license: str
    sim: SimSpec
    real: RealSpec = Field(default_factory=RealSpec)
    available: bool = True
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if not self.id:
            raise ValueError("Robot id must be non-empty")
        if not self.available and not self.unavailable_reason:
            raise ValueError("Unavailable robots require unavailable_reason")
        if (
            self.available
            and self.sim.source is SimSource.MENAGERIE
            and (not self.sim.menagerie_dir or not self.sim.xml_file)
        ):
            raise ValueError("Menagerie robots require menagerie_dir and xml_file")
        return self
