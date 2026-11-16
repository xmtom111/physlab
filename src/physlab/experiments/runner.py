"""Execute a protocol on a factorial grid.

Trials run in an order shuffled by ``numpy.random.Generator(protocol.seed)``.
A skill that raises is recorded with its message. The trial stays in the
table.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np

from physlab.chem.virtual_lab import ChemReading, VirtualChemLab
from physlab.core.errors import ConfigError, PhysLabError
from physlab.core.events import Actor, EventLog
from physlab.experiments.protocol import ExperimentProtocol, validate_protocol
from physlab.experiments.registry import SkillRegistry
from physlab.experiments.results import RunResult, TrialContext, TrialRecord
from physlab.experiments.storage import write_protocol, write_trials


class ExperimentRunner:
    """Run protocols and keep every trial."""

    def __init__(
        self,
        event_log: EventLog,
        world: object | None = None,
        robot: object | None = None,
        devices: object | None = None,
        skills: SkillRegistry | None = None,
    ) -> None:
        """Store collaborators.

        Args:
            event_log: Log that receives one event per trial. Its directory
                is ``runs/<run_id>``.
            world: Optional simulator passed through to skills.
            robot: Optional robot passed through to skills.
            devices: Optional device bundle passed through to skills.
            skills: Registry. The experiment registry is used when omitted.
        """
        self.event_log = event_log
        self.world = world
        self.robot = robot
        self.devices = devices
        self.skills = skills if skills is not None else _default_skills()

    def run(self, protocol: ExperimentProtocol) -> RunResult:
        """Execute the factorial grid.

        Args:
            protocol: Validated against the skill registry before any trial.

        Returns:
            Trials in shuffled order, with failures kept.

        Raises:
            ConfigError: The protocol names an unknown skill or a duplicate
                variable.
        """
        issues = validate_protocol(protocol, self.skills.names())
        if issues:
            raise ConfigError("Experiment protocol is invalid", details={"issues": issues})
        directory = self.event_log.path.parent
        write_protocol(directory, protocol)
        trials = [
            _run_one(self, protocol, values, rep, order)
            for order, (values, rep) in enumerate(_order(protocol))
        ]
        write_trials(directory, trials)
        return RunResult(
            run_id=self.event_log.run_id,
            directory=directory,
            protocol_id=protocol.id,
            question=protocol.question,
            hypothesis=protocol.hypothesis,
            trials=trials,
        )

    def run_chem(
        self,
        protocol: object,
        *,
        question: str,
        hypothesis: str,
        seed: int = 0,
        analyte: str = "food_dye",
        lab: VirtualChemLab | None = None,
        reset: bool = True,
    ) -> RunResult:
        """Run a chemistry protocol and store one trial per reading.

        Args:
            protocol: A :class:`physlab.chem.protocol.ChemProtocol`. Typed as
                ``object`` here so importing the runner does not load the
                chemistry stack until a chem trial is requested.
            question: Stored on the run result for the report.
            hypothesis: Stored on the run result for the report.
            seed: Plate-reader seed. The pH probe uses ``seed + 1``.
            analyte: Species recorded as ``concentration_m``.
            lab: Optional persistent chemistry lab.
            reset: Reload the supplied lab.

        Returns:
            Trials whose measurements are absorbance or pH. ``fit_models``
            can take the resulting table.

        Raises:
            ConfigError: Volumes or tips do not fit.
            SafetyViolation: The reagent allowlist denies the protocol.
        """
        from physlab.chem.protocol import ChemProtocol
        from physlab.chem.virtual_lab import VirtualChemLab

        if not isinstance(protocol, ChemProtocol):
            raise ConfigError("run_chem requires a ChemProtocol")
        directory = self.event_log.path.parent
        (directory / "media").mkdir(parents=True, exist_ok=True)
        text = protocol.model_dump_json(indent=2)
        (directory / "protocol.json").write_text(text, encoding="utf-8")
        readings = (lab or VirtualChemLab(seed=seed, analyte=analyte)).execute(
            protocol, reset=True if lab is None else reset
        )
        trials = _chem_trials(readings)
        write_trials(directory, trials)
        self.event_log.record(
            Actor.SYSTEM,
            "chem_protocol",
            params={"protocol_id": protocol.id, "seed": seed},
            result={"trials": len(trials)},
        )
        return RunResult(
            run_id=self.event_log.run_id,
            directory=directory,
            protocol_id=protocol.id,
            question=question,
            hypothesis=hypothesis,
            trials=trials,
        )


def _chem_trials(readings: Sequence[ChemReading]) -> list[TrialRecord]:
    trials: list[TrialRecord] = []
    for index, reading in enumerate(readings):
        if not isinstance(reading, ChemReading):
            continue
        trials.append(
            TrialRecord(
                order=index,
                repetition=0,
                values=dict(reading.values),
                measurements=dict(reading.measurements),
                success=True,
                reason="",
            )
        )
    return trials


def _default_skills() -> SkillRegistry:
    from physlab.experiments.registry import experiment_registry

    return experiment_registry()


def _order(protocol: ExperimentProtocol) -> list[tuple[dict[str, float], int]]:
    names = [item.name for item in protocol.independent]
    levels = [item.values for item in protocol.independent]
    grid: list[tuple[dict[str, float], int]] = []
    for combo in itertools.product(*levels) if levels else [()]:
        assignment = dict(zip(names, combo, strict=True))
        for rep in range(protocol.repetitions):
            grid.append((assignment, rep))
    generator = np.random.default_rng(protocol.seed)
    generator.shuffle(grid)
    return grid


def _run_one(
    runner: ExperimentRunner,
    protocol: ExperimentProtocol,
    assignment: dict[str, float],
    repetition: int,
    order: int,
) -> TrialRecord:
    values = {**protocol.controlled, **assignment}
    ctx = TrialContext(values, runner.world, runner.robot, runner.devices)
    reason = _execute(runner, protocol, ctx)
    measurements = dict(ctx.records) if not reason else {}
    if not reason:
        reason = _missing(protocol, ctx)
    record = TrialRecord(
        order=order,
        repetition=repetition,
        values=values,
        measurements=measurements,
        success=not reason,
        reason=reason,
    )
    runner.event_log.record(
        Actor.SYSTEM,
        "experiment_trial",
        params={"order": order, "repetition": repetition, **values},
        result={"success": record.success, "reason": reason, **measurements},
    )
    return record


def _execute(runner: ExperimentRunner, protocol: ExperimentProtocol, ctx: TrialContext) -> str:
    try:
        for step in protocol.procedure:
            ctx.step_args = dict(step.args)
            runner.skills.get(step.skill)(ctx)
    except (PhysLabError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        message = exc.message if isinstance(exc, PhysLabError) else str(exc)
        return message
    return ""


def _missing(protocol: ExperimentProtocol, ctx: TrialContext) -> str:
    missing = [item.name for item in protocol.dependent if item.name not in ctx.records]
    if missing:
        return "missing measurement " + ", ".join(missing)
    return ""
