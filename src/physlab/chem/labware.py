"""OT-2 style labware and deck.

Slots are ``"1"`` through ``"11"``. A 96-well plate holds 300 µL per well.
Tube racks, the 12-well reservoir, and a 300 µL tip rack fill the rest of
the catalog.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from physlab.core.errors import ConfigError

DECK_SLOTS: tuple[str, ...] = tuple(str(index) for index in range(1, 12))
PIPETTE_MAX_UL = 300.0


class LabwareKind(StrEnum):
    """Labware a deck slot can hold."""

    WELL_PLATE_96 = "well_plate_96"
    TUBE_RACK_15ML = "tube_rack_15ml"
    TUBE_RACK_50ML = "tube_rack_50ml"
    RESERVOIR_12 = "reservoir_12"
    TIPRACK_300UL = "tiprack_300ul"


class Well(BaseModel):
    """One well or tube.

    Attributes:
        id: Well name such as ``A1``.
        max_volume_ul: Capacity, microliters.
        contents: Moles of each species. The solvent volume is ``volume_ul``.
        volume_ul: Current liquid volume, microliters.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    id: str
    max_volume_ul: float = Field(gt=0)
    contents: dict[str, float] = Field(default_factory=dict)
    volume_ul: float = Field(default=0.0, ge=0)


class Labware(BaseModel):
    """One item on a deck slot.

    Attributes:
        kind: Catalog name.
        wells: Liquid wells. Empty for a tip rack.
        tips_remaining: Unused tips. Zero for liquid labware.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    kind: LabwareKind
    wells: dict[str, Well]
    tips_remaining: int = Field(ge=0)


class Deck(BaseModel):
    """Slots ``1`` through ``11``.

    Attributes:
        slots: Labware keyed by slot. Missing slots are empty.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    slots: dict[str, Labware]


def empty_labware(kind: LabwareKind) -> Labware:
    """Build an empty piece of labware.

    Args:
        kind: Catalog name.

    Returns:
        Labware with empty wells, or a full tip rack.
    """
    if kind is LabwareKind.TIPRACK_300UL:
        return Labware(kind=kind, wells={}, tips_remaining=96)
    volume, names = _CATALOG[kind]
    wells = {name: Well(id=name, max_volume_ul=volume) for name in names}
    return Labware(kind=kind, wells=wells, tips_remaining=0)


def parse_location(location: str) -> tuple[str, str]:
    """Split ``slot:well``.

    Args:
        location: Location such as ``2:A1``.

    Returns:
        Slot and well id.

    Raises:
        ConfigError: The location is not ``slot:well`` on the OT-2 deck.
    """
    slot, separator, well = location.partition(":")
    if separator != ":" or slot not in DECK_SLOTS or not well:
        raise ConfigError(
            "Location must be slot:well on deck slots 1-11", details={"location": location}
        )
    return slot, well


def _wells(rows: int, columns: int) -> tuple[str, ...]:
    names: list[str] = []
    for row in range(rows):
        for column in range(1, columns + 1):
            names.append(f"{chr(ord('A') + row)}{column}")
    return tuple(names)


_CATALOG: dict[LabwareKind, tuple[float, tuple[str, ...]]] = {
    LabwareKind.WELL_PLATE_96: (300.0, _wells(8, 12)),
    LabwareKind.TUBE_RACK_15ML: (15_000.0, _wells(3, 5)),
    LabwareKind.TUBE_RACK_50ML: (50_000.0, _wells(3, 2)),
    LabwareKind.RESERVOIR_12: (15_000.0, _wells(1, 12)),
}
