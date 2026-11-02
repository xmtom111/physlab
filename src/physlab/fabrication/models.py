"""Shared fabrication records.

Lengths on a :class:`PartArtifact` are meters. CAD parameters stay in the
units of the part schema, which are millimeters.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from physlab.core.types import Pose
from physlab.sim.scene import PrimitiveSpec


class PartArtifact(BaseModel):
    """One generated part, ready to check and print.

    Attributes:
        name: Library name.
        params: Parameter values that produced the solid.
        stl_path: Triangle mesh in millimeters, for slicing.
        step_path: STEP file in millimeters.
        bbox_m: Axis-aligned size, meters, X Y Z.
        collision_primitives: Coarse colliders in the part frame.
        frames: Named mating frames in the part frame (``frame="body"``).
        aruco_face: Pose of the marker recess, Z out of the face, or ``None``.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    name: str
    params: dict[str, float | int | str]
    stl_path: Path
    step_path: Path
    bbox_m: tuple[float, float, float]
    collision_primitives: list[PrimitiveSpec]
    frames: dict[str, Pose]
    aruco_face: Pose | None = None


class ListedPart(BaseModel):
    """A library name and the JSON schema of its parameters."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    name: str
    json_schema: dict[str, object]


class PrintState(StrEnum):
    """Lifecycle of one print job."""

    QUEUED = "QUEUED"
    SLICING = "SLICING"
    PRINTING = "PRINTING"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class PrintStatus(BaseModel):
    """Progress reported by a print service.

    Attributes:
        state: Lifecycle state.
        progress: Fraction complete, from 0 to 1.
        message: Short status text.
        eta_s: Remaining seconds, or ``None`` when unknown.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    state: PrintState
    progress: float = Field(ge=0, le=1)
    message: str = ""
    eta_s: float | None = None


class PrintJob(BaseModel):
    """A request to print one artifact.

    Attributes:
        part: Geometry to print.
        material: Filament name.
        quality: Profile name understood by the service.
        copies: How many copies to print.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    part: PartArtifact
    material: str = "PLA"
    quality: str = "standard"
    copies: int = Field(default=1, ge=1)


class InventoryItem(BaseModel):
    """One physical copy tracked after a print finishes.

    Attributes:
        part_instance_id: Stable id of this copy.
        part_name: Library name.
        aruco_id: Marker id assigned in simulation, or ``None``.
        location: Where the copy is. ``print_pickup_zone`` is the sim slot.
        pose: Body pose when the copy was spawned, or ``None``.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    part_instance_id: str
    part_name: str
    aruco_id: int | None = None
    location: str
    pose: Pose | None = None


class FabricationResult(BaseModel):
    """Outcome of :func:`physlab.fabrication.print_service.fabricate`.

    Attributes:
        ok: True when the part was printable and the job finished ``DONE``.
        part: Built artifact, including when the print check failed.
        job_id: Service job id, or ``None`` when the check stopped the pipeline.
        status: Terminal status, or ``None`` when the job was not submitted.
        inventory: Copy recorded for a finished job, or ``None``.
        issues: Reasons the pipeline stopped.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    ok: bool
    part: PartArtifact | None = None
    job_id: str | None = None
    status: PrintStatus | None = None
    inventory: InventoryItem | None = None
    issues: list[str] = Field(default_factory=list)
