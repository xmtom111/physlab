"""Declarative experiment protocols.

A protocol names the question, the variables, and the skills to run. The
runner rejects a protocol whose procedure names a skill the registry does
not know.
"""

from __future__ import annotations

from collections.abc import Collection
from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class SafetyClass(StrEnum):
    """How much human approval a protocol needs."""

    SIM_ONLY = "sim_only"
    LOW = "low"
    NEEDS_APPROVAL = "needs_approval"


class Variable(BaseModel):
    """One independent variable swept on a grid.

    Attributes:
        name: Column name recorded on each trial.
        unit: SI unit.
        values: Levels. The runner crosses these with the other variables.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    values: list[float] = Field(min_length=1)


class Measurement(BaseModel):
    """One dependent quantity a trial must record.

    Attributes:
        name: Column name.
        unit: SI unit.
        method: How the value is obtained, for the report.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    method: str = Field(min_length=1)


class ProcedureStep(BaseModel):
    """One skill invocation inside a trial.

    Attributes:
        skill: Name in the skill registry.
        args: Numeric arguments. Sweep values are already on the trial.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    skill: str = Field(min_length=1)
    args: dict[str, float] = Field(default_factory=dict)


class ExperimentProtocol(BaseModel):
    """A reproducible experiment the runner can execute.

    Attributes:
        id: Stable name.
        question: What the experiment asks.
        hypothesis: What the experimenter expects.
        apparatus: Scene or plan the procedure assumes.
        independent: Variables on the factorial grid.
        dependent: Measurements each trial must produce.
        controlled: Constants copied onto every trial.
        procedure: Skills, in order, for one trial.
        repetitions: Repeats of each grid point. At least one.
        seed: Seed for the trial shuffle.
        safety_class: Sim-only, low, or needs approval.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    apparatus: str = Field(min_length=1)
    independent: list[Variable] = Field(default_factory=list)
    dependent: list[Measurement] = Field(min_length=1)
    controlled: dict[str, float] = Field(default_factory=dict)
    procedure: list[ProcedureStep] = Field(min_length=1)
    repetitions: int = Field(ge=1)
    seed: int
    safety_class: SafetyClass


def validate_protocol(protocol: ExperimentProtocol, known_skills: Collection[str]) -> list[str]:
    """Return validation issues. An empty list means the protocol can run.

    Args:
        protocol: Protocol to check.
        known_skills: Skill names the registry can call.

    Returns:
        Human-readable issues. Unknown procedure skills are included.
    """
    issues: list[str] = []
    known = set(known_skills)
    seen = _names(protocol, issues)
    for step in protocol.procedure:
        if step.skill not in known:
            issues.append(f"unknown skill {step.skill}")
    for name in seen:
        del name
    return issues


def _names(protocol: ExperimentProtocol, issues: list[str]) -> set[str]:
    seen: set[str] = set()
    for variable in protocol.independent:
        _claim(seen, issues, variable.name, "independent")
    for measurement in protocol.dependent:
        _claim(seen, issues, measurement.name, "dependent")
    for name in protocol.controlled:
        _claim(seen, issues, name, "controlled")
    return seen


def _claim(seen: set[str], issues: list[str], name: str, kind: str) -> None:
    if name in seen:
        issues.append(f"duplicate name {name} ({kind})")
    seen.add(name)
