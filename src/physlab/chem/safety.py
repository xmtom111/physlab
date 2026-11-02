"""Load the human reagent allowlist and check a protocol against it.

``configs/chem/reagents.yaml`` is read only. Nothing in this package opens
it for writing.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from physlab.chem.protocol import ChemProtocol, IncubateStep, TransferStep
from physlab.core.errors import ConfigError
from physlab.core.types import Mode

SUPPORTED_STEPS = frozenset({"transfer", "mix", "incubate", "read_absorbance", "read_ph"})


class ReagentSpec(BaseModel):
    """One allowlisted reagent.

    Attributes:
        name: Display name.
        cas: CAS number, or ``generic`` when the dye is not a single substance.
        hazard_codes: GHS hazard statements.
        hazard_class: Class used for incompatibility.
        max_concentration_m: Maximum molarity that may be placed on the deck.
        max_total_volume_ml: Maximum combined fill volume, milliliters.
        incompatible_classes: Hazard classes that must not share a well.
        strong_base: True when the solute is a fully dissociated hydroxide.
        ka: Acid dissociation constant. ``None`` when the solute is not a weak acid.
        epsilon_l_mol_cm: Molar absorptivity keyed by wavelength in nanometers.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    name: str
    cas: str
    hazard_codes: list[str]
    hazard_class: str
    max_concentration_m: float = Field(gt=0)
    max_total_volume_ml: float = Field(gt=0)
    incompatible_classes: list[str]
    strong_base: bool = False
    ka: float | None = None
    epsilon_l_mol_cm: dict[int, float] = Field(default_factory=dict)


class KineticPair(BaseModel):
    """A first-order conversion the virtual lab is allowed to run.

    Attributes:
        reactant: Allowlist species consumed.
        product: Allowlist species produced.
        k_per_s: Rate constant used when Arrhenius parameters are absent.
        a_per_s: Arrhenius prefactor. Used with ``ea_j_mol``.
        ea_j_mol: Activation energy, joules per mole.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    reactant: str
    product: str
    k_per_s: float = 0.0
    a_per_s: float | None = None
    ea_j_mol: float | None = None


class Allowlist(BaseModel):
    """Parsed reagent file.

    Attributes:
        max_temp_c: Hottest incubate step that is allowed.
        path_cm: Optical path used by the virtual plate reader.
        kinetics: First-order pairs. Both species must be allowlisted.
        reagents: Reagents keyed by id.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    max_temp_c: float
    path_cm: float = Field(gt=0)
    kinetics: list[KineticPair]
    reagents: dict[str, ReagentSpec]


class SafetyDecision(BaseModel):
    """Result of checking one protocol.

    Attributes:
        allowed: True when ``violations`` is empty.
        violations: Reasons the protocol is denied.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    allowed: bool
    violations: list[str]


class ChemSafetyPolicy:
    """Checks protocols against the on-disk allowlist."""

    def __init__(self, allowlist: Allowlist) -> None:
        """Store a parsed allowlist.

        Args:
            allowlist: Contents of the reagent file.
        """
        self.allowlist = allowlist

    @classmethod
    def load(cls, path: Path | None = None) -> ChemSafetyPolicy:
        """Read the reagent file. This never writes it.

        Args:
            path: Explicit file. The checkout's ``configs/chem/reagents.yaml``
                is used when omitted.

        Returns:
            Policy ready to check protocols.

        Raises:
            ConfigError: The file is missing or does not match the schema.
        """
        source = _reagents_path() if path is None else path
        if not source.is_file():
            raise ConfigError("Reagent allowlist was not found", details={"path": str(source)})
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
        try:
            allowlist = Allowlist.model_validate(loaded)
        except ValidationError as exc:
            raise ConfigError(
                "Reagent allowlist is invalid", details={"path": str(source)}
            ) from exc
        return cls(allowlist)

    def check_protocol(
        self,
        protocol: ChemProtocol,
        *,
        mode: Mode = Mode.SIM,
        approval_token: str | None = None,
    ) -> SafetyDecision:
        """Decide whether a protocol may run.

        Args:
            protocol: Protocol to check.
            mode: ``real`` additionally requires an approval token. The token
                is not issued here; T12 owns that gate.
            approval_token: Human approval token. Ignored in simulation.

        Returns:
            Allowed flag and every violation.
        """
        violations = _reagent_violations(self.allowlist, protocol)
        violations.extend(_step_violations(self.allowlist, protocol))
        violations.extend(_mixture_violations(self.allowlist, protocol))
        if mode is Mode.REAL and not approval_token:
            violations.append("real mode requires an approval token")
        return SafetyDecision(allowed=not violations, violations=violations)


def _reagent_violations(allowlist: Allowlist, protocol: ChemProtocol) -> list[str]:
    totals: dict[str, float] = {}
    violations: list[str] = []
    for location, fill in protocol.reagents.items():
        spec = allowlist.reagents.get(fill.reagent)
        if spec is None:
            violations.append(f"unknown reagent {fill.reagent} at {location}")
            continue
        if fill.concentration_m - spec.max_concentration_m > 1e-12:
            violations.append(
                f"{fill.reagent} at {location} is {fill.concentration_m} M, "
                f"above {spec.max_concentration_m} M"
            )
        totals[fill.reagent] = totals.get(fill.reagent, 0.0) + fill.volume_ul
    for name, volume_ul in totals.items():
        spec = allowlist.reagents[name]
        limit_ul = spec.max_total_volume_ml * 1000.0
        if volume_ul - limit_ul > 1e-6:
            violations.append(f"{name} volume {volume_ul} uL exceeds {limit_ul} uL")
    return violations


def _step_violations(allowlist: Allowlist, protocol: ChemProtocol) -> list[str]:
    violations: list[str] = []
    for step in protocol.steps:
        if step.kind not in SUPPORTED_STEPS:
            violations.append(f"unsupported step {step.kind}")
        if isinstance(step, IncubateStep) and step.temp_c - allowlist.max_temp_c > 1e-9:
            violations.append(f"incubate at {step.temp_c} C exceeds {allowlist.max_temp_c} C")
    return violations


def _mixture_violations(allowlist: Allowlist, protocol: ChemProtocol) -> list[str]:
    present: dict[str, set[str]] = {}
    for location, fill in protocol.reagents.items():
        if fill.concentration_m > 0.0:
            present.setdefault(location, set()).add(fill.reagent)
    for step in protocol.steps:
        if isinstance(step, TransferStep):
            moved = present.get(step.src, set())
            present.setdefault(step.dst, set()).update(moved)
    violations: list[str] = []
    for location, species in present.items():
        violations.extend(_pair_violations(allowlist, location, species))
    return violations


def _pair_violations(allowlist: Allowlist, location: str, species: set[str]) -> list[str]:
    names = sorted(species)
    violations: list[str] = []
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if _clash(allowlist, left, right):
                violations.append(f"incompatible mixture of {left} and {right} in {location}")
    return violations


def _clash(allowlist: Allowlist, left: str, right: str) -> bool:
    a = allowlist.reagents.get(left)
    b = allowlist.reagents.get(right)
    if a is None or b is None:
        return False
    return b.hazard_class in a.incompatible_classes or a.hazard_class in b.incompatible_classes


def _reagents_path() -> Path:
    module = Path(__file__).resolve()
    candidate = module.parents[3] / "configs" / "chem" / "reagents.yaml"
    if candidate.is_file():
        return candidate
    cwd = Path.cwd().resolve()
    for directory in (cwd, *cwd.parents):
        found = directory / "configs" / "chem" / "reagents.yaml"
        if found.is_file():
            return found
    return candidate
