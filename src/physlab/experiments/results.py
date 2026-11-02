"""Trial and run records shared by the runner, storage, and reports."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class TrialContext:
    """Values and measurements for one trial.

    Attributes:
        values: Controlled constants plus this trial's independent levels.
        records: Measurements written by skills.
        world: Optional simulator owned by the runner.
        robot: Optional robot owned by the runner.
        devices: Optional device bundle owned by the runner.
        step_args: Arguments of the skill currently running.
    """

    def __init__(
        self,
        values: dict[str, float],
        world: object | None,
        robot: object | None,
        devices: object | None,
    ) -> None:
        """Store the trial inputs.

        Args:
            values: Numeric inputs for the skills.
            world: Simulator, or ``None`` when the plant builds its own.
            robot: Robot, or ``None``.
            devices: Device bundle, or ``None``.
        """
        self.values = values
        self.records: dict[str, float] = {}
        self.world = world
        self.robot = robot
        self.devices = devices
        self.step_args: dict[str, float] = {}


class TrialRecord(BaseModel):
    """One grid point, successful or not.

    Attributes:
        order: Position after the seeded shuffle. Zero is first.
        repetition: Repeat index before the shuffle.
        values: Independent and controlled numbers.
        measurements: Dependent numbers. Empty when the trial failed early.
        success: True when every skill finished and every measurement exists.
        reason: Empty on success, otherwise the error message.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    order: int
    repetition: int
    values: dict[str, float]
    measurements: dict[str, float] = Field(default_factory=dict)
    success: bool
    reason: str = ""


class RunResult(BaseModel):
    """A finished protocol, including failed trials.

    Attributes:
        run_id: Event-log id. The directory is ``runs/<run_id>``.
        directory: Run directory.
        protocol_id: Protocol id.
        question: Protocol question, copied for the report.
        hypothesis: Protocol hypothesis, copied for the report.
        trials: Every trial in execution order.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    directory: Path
    protocol_id: str
    question: str
    hypothesis: str
    trials: list[TrialRecord]
