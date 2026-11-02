"""Execute a chemistry protocol on ideal liquid handling and equilibrium.

Transfers scale every solute with the volume fraction. pH and absorbance
come from :mod:`physlab.chem.reactions`. Instrument noise is seeded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from physlab.chem.instruments import PhProbe, PlateReader
from physlab.chem.labware import empty_labware
from physlab.chem.protocol import (
    ChemProtocol,
    ChemStep,
    IncubateStep,
    MixStep,
    ReadAbsorbanceStep,
    ReadPhStep,
    TransferStep,
    require_valid,
)
from physlab.chem.reactions import (
    absorbance_of,
    advance_kinetics,
    moles_from_fill,
    ph_of,
    transfer_contents,
)
from physlab.chem.safety import ChemSafetyPolicy
from physlab.core.errors import SafetyViolation
from physlab.core.types import Mode


@dataclass
class ChemReading:
    """One instrument sample, shaped for the experiment runner.

    Attributes:
        values: Independent numbers, such as concentration or titrant volume.
        measurements: Dependent numbers, such as absorbance or pH.
    """

    values: dict[str, float]
    measurements: dict[str, float]


@dataclass
class _LiveWell:
    max_volume_ul: float
    contents: dict[str, float] = field(default_factory=dict)
    volume_ul: float = 0.0


class VirtualChemLab:
    """In-memory deck that runs one protocol."""

    def __init__(
        self,
        policy: ChemSafetyPolicy | None = None,
        *,
        seed: int = 0,
        mode: Mode = Mode.SIM,
        approval_token: str | None = None,
        analyte: str = "food_dye",
    ) -> None:
        """Create the deck instruments.

        Args:
            policy: Allowlist. The checkout file is loaded when omitted.
            seed: Noise seed for the plate reader. The pH probe uses ``seed + 1``.
            mode: Passed to the safety check.
            approval_token: Required when ``mode`` is real.
            analyte: Species whose molarity is recorded on absorbance readings.
        """
        self.policy = ChemSafetyPolicy.load() if policy is None else policy
        self.reader = PlateReader(seed)
        self.probe = PhProbe(seed + 1)
        self.mode = mode
        self.approval_token = approval_token
        self.analyte = analyte
        self._wells: dict[str, _LiveWell] = {}
        self._stocks: dict[str, str] = {}
        self._titrant_ul: dict[str, float] = {}

    def execute(self, protocol: ChemProtocol) -> list[ChemReading]:
        """Run every step.

        Args:
            protocol: Already described deck, fills, and steps.

        Returns:
            One reading per absorbance well and per pH read.

        Raises:
            ConfigError: Volumes or tips do not fit.
            SafetyViolation: The allowlist denies the protocol.
        """
        require_valid(protocol)
        decision = self.policy.check_protocol(
            protocol,
            mode=self.mode,
            approval_token=self.approval_token,
        )
        if not decision.allowed:
            raise SafetyViolation(
                "Chemistry protocol denied", details={"violations": decision.violations}
            )
        self._load(protocol)
        readings: list[ChemReading] = []
        for step in protocol.steps:
            readings.extend(self._apply(protocol, step))
        return readings

    def _load(self, protocol: ChemProtocol) -> None:
        self._wells = {}
        self._stocks = {}
        self._titrant_ul = {}
        for slot, kind in protocol.labware.items():
            for name, template in empty_labware(kind).wells.items():
                location = f"{slot}:{name}"
                self._wells[location] = _LiveWell(max_volume_ul=template.max_volume_ul)
        for location, fill in protocol.reagents.items():
            live = self._wells[location]
            live.volume_ul += fill.volume_ul
            if fill.concentration_m > 0.0:
                added = moles_from_fill(fill.concentration_m, fill.volume_ul)
                live.contents[fill.reagent] = live.contents.get(fill.reagent, 0.0) + added
            self._stocks[location] = fill.reagent

    def _apply(self, protocol: ChemProtocol, step: ChemStep) -> list[ChemReading]:
        del protocol
        if isinstance(step, TransferStep):
            self._transfer(step)
        elif isinstance(step, IncubateStep):
            self._incubate(step)
        elif isinstance(step, ReadAbsorbanceStep):
            return [self._absorbance(location, step.wavelength_nm) for location in step.wells]
        elif isinstance(step, ReadPhStep):
            return [self._ph(step.well)]
        elif isinstance(step, MixStep):
            return []
        return []

    def _transfer(self, step: TransferStep) -> None:
        source = self._wells[step.src]
        moved = transfer_contents(source.contents, source.volume_ul, step.volume_ul)
        for name, amount in moved.items():
            source.contents[name] = source.contents.get(name, 0.0) - amount
        source.volume_ul -= step.volume_ul
        dest = self._wells[step.dst]
        for name, amount in moved.items():
            dest.contents[name] = dest.contents.get(name, 0.0) + amount
        dest.volume_ul += step.volume_ul
        if self._stocks.get(step.src) == "sodium_hydroxide":
            self._titrant_ul[step.dst] = self._titrant_ul.get(step.dst, 0.0) + step.volume_ul

    def _incubate(self, step: IncubateStep) -> None:
        allowed = set(self.policy.allowlist.reagents)
        for well in self._wells.values():
            well.contents = advance_kinetics(
                well.contents,
                step.duration_s,
                step.temp_c,
                self.policy.allowlist.kinetics,
                allowed,
            )

    def _absorbance(self, location: str, wavelength_nm: float) -> ChemReading:
        well = self._wells[location]
        wavelength = int(wavelength_nm)
        ideal = absorbance_of(well.contents, well.volume_ul, wavelength, self.policy.allowlist)
        concentration = _molarity(well, self.analyte)
        return ChemReading(
            values={"concentration_m": concentration},
            measurements={"absorbance": self.reader.read(ideal)},
        )

    def _ph(self, location: str) -> ChemReading:
        well = self._wells[location]
        ideal = ph_of(well.contents, well.volume_ul, self.policy.allowlist)
        return ChemReading(
            values={"titrant_ul": self._titrant_ul.get(location, 0.0)},
            measurements={"ph": self.probe.read(ideal)},
        )


def _molarity(well: _LiveWell, analyte: str) -> float:
    if well.volume_ul <= 0.0:
        return 0.0
    return well.contents.get(analyte, 0.0) / (well.volume_ul * 1.0e-6)
