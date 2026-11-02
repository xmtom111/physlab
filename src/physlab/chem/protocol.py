"""Chemistry protocols and the checks that do not depend on hazard policy.

Volumes must fit the labware and the deck must hold enough 300 µL tips.
Hazard checks live in :class:`physlab.chem.safety.ChemSafetyPolicy`.
"""

from __future__ import annotations

import math
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from physlab.chem.labware import (
    DECK_SLOTS,
    PIPETTE_MAX_UL,
    LabwareKind,
    empty_labware,
    parse_location,
)
from physlab.core.errors import ConfigError


class ReagentFill(BaseModel):
    """Solution placed in a well before the first step.

    Attributes:
        reagent: Allowlist name.
        concentration_m: Molarity. Zero is pure solvent.
        volume_ul: Volume placed, microliters.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    reagent: str = Field(min_length=1)
    concentration_m: float = Field(ge=0)
    volume_ul: float = Field(gt=0)


class TransferStep(BaseModel):
    """Move liquid from one well to another.

    Attributes:
        kind: Always ``transfer``.
        src: Source ``slot:well``.
        dst: Destination ``slot:well``.
        volume_ul: Volume moved, microliters.
        mix_after: Mix the destination after the liquid arrives.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["transfer"] = "transfer"
    src: str
    dst: str
    volume_ul: float = Field(gt=0)
    mix_after: bool = False


class MixStep(BaseModel):
    """Mix a well in place.

    Attributes:
        kind: Always ``mix``.
        well: ``slot:well``.
        volume_ul: Volume of each mix cycle, microliters.
        cycles: Number of mix cycles.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["mix"] = "mix"
    well: str
    volume_ul: float = Field(gt=0)
    cycles: int = Field(ge=1)


class IncubateStep(BaseModel):
    """Hold the deck at a temperature.

    Attributes:
        kind: Always ``incubate``.
        duration_s: Seconds.
        temp_c: Celsius.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["incubate"] = "incubate"
    duration_s: float = Field(gt=0)
    temp_c: float


class ReadAbsorbanceStep(BaseModel):
    """Read wells at one wavelength.

    Attributes:
        kind: Always ``read_absorbance``.
        wells: ``slot:well`` locations.
        wavelength_nm: Wavelength, nanometers.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["read_absorbance"] = "read_absorbance"
    wells: list[str] = Field(min_length=1)
    wavelength_nm: float = Field(gt=0)


class ReadPhStep(BaseModel):
    """Read the pH of one well.

    Attributes:
        kind: Always ``read_ph``.
        well: ``slot:well``.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["read_ph"] = "read_ph"
    well: str


ChemStep = Annotated[
    TransferStep | MixStep | IncubateStep | ReadAbsorbanceStep | ReadPhStep,
    Field(discriminator="kind"),
]


class ChemProtocol(BaseModel):
    """One liquid-handling protocol.

    Attributes:
        id: Stable name.
        labware: Deck slot to labware kind.
        reagents: Initial fills keyed by ``slot:well``.
        steps: Ordered liquid-handling and measurement steps.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    labware: dict[str, LabwareKind]
    reagents: dict[str, ReagentFill]
    steps: list[ChemStep] = Field(min_length=1)


def validate_chem_protocol(protocol: ChemProtocol) -> list[str]:
    """Return volume and tip issues. An empty list means the deck can run it.

    Args:
        protocol: Protocol to check.

    Returns:
        Human-readable issues.
    """
    issues = _deck_issues(protocol)
    if issues:
        return issues
    return _simulate_volumes(protocol)


def require_valid(protocol: ChemProtocol) -> None:
    """Raise when volumes or tips do not fit.

    Args:
        protocol: Protocol to check.

    Raises:
        ConfigError: Validation returned any issue.
    """
    issues = validate_chem_protocol(protocol)
    if issues:
        raise ConfigError("Chemistry protocol is invalid", details={"issues": issues})


def _deck_issues(protocol: ChemProtocol) -> list[str]:
    issues: list[str] = []
    for slot in protocol.labware:
        if slot not in DECK_SLOTS:
            issues.append(f"slot {slot} is outside 1-11")
    for location in protocol.reagents:
        issues.extend(_well_issues(protocol, location, "fill"))
    for step in protocol.steps:
        issues.extend(_step_locations(protocol, step))
    return issues


def _step_locations(protocol: ChemProtocol, step: ChemStep) -> list[str]:
    if isinstance(step, TransferStep):
        return _well_issues(protocol, step.src, "source") + _well_issues(
            protocol, step.dst, "destination"
        )
    if isinstance(step, MixStep | ReadPhStep):
        return _well_issues(protocol, step.well, "well")
    if isinstance(step, ReadAbsorbanceStep):
        issues: list[str] = []
        for location in step.wells:
            issues.extend(_well_issues(protocol, location, "well"))
        return issues
    return []


def _well_issues(protocol: ChemProtocol, location: str, role: str) -> list[str]:
    try:
        slot, well = parse_location(location)
    except ConfigError:
        return [f"{role} {location} is not slot:well"]
    labware = protocol.labware.get(slot)
    if labware is None:
        return [f"{role} {location} has no labware"]
    if labware is LabwareKind.TIPRACK_300UL:
        return [f"{role} {location} is a tip rack"]
    if well not in empty_labware(labware).wells:
        return [f"{role} {location} is not on that labware"]
    return []


def _simulate_volumes(protocol: ChemProtocol) -> list[str]:
    volumes = {location: fill.volume_ul for location, fill in protocol.reagents.items()}
    issues: list[str] = []
    tips = 0
    for step in protocol.steps:
        if isinstance(step, TransferStep):
            tips += _tips_for(step.volume_ul)
            issues.extend(_move(protocol, volumes, step))
        elif isinstance(step, MixStep):
            issues.extend(_fits(protocol, step.well, step.volume_ul, volumes))
    capacity = sum(96 for kind in protocol.labware.values() if kind is LabwareKind.TIPRACK_300UL)
    if tips > capacity:
        issues.append(f"protocol needs {tips} tips but the deck has {capacity}")
    return issues


def _move(protocol: ChemProtocol, volumes: dict[str, float], step: TransferStep) -> list[str]:
    source = volumes.get(step.src, 0.0)
    if step.volume_ul - source > 1e-9:
        return [f"transfer of {step.volume_ul} uL from {step.src} exceeds its {source} uL"]
    volumes[step.src] = source - step.volume_ul
    volumes[step.dst] = volumes.get(step.dst, 0.0) + step.volume_ul
    return _fits(protocol, step.dst, volumes[step.dst], volumes)


def _fits(
    protocol: ChemProtocol, location: str, volume_ul: float, volumes: dict[str, float]
) -> list[str]:
    del volumes
    slot, well = parse_location(location)
    kind = protocol.labware[slot]
    capacity = empty_labware(kind).wells[well].max_volume_ul
    if volume_ul - capacity > 1e-6:
        return [f"{location} holds {capacity} uL but the protocol puts {volume_ul} uL in it"]
    return []


def _tips_for(volume_ul: float) -> int:
    return max(1, math.ceil(volume_ul / PIPETTE_MAX_UL - 1e-9))
