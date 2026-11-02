"""Fits, the runner, and the three library experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from physlab.core.errors import ConfigError
from physlab.core.events import EventLog
from physlab.experiments.analysis import (
    ModelFit,
    compile_expression,
    fit_custom,
    fit_models,
    report,
)
from physlab.experiments.library.incline import SIM_FRICTION, incline_protocol
from physlab.experiments.library.lens import TRUE_FOCAL_M, lens_protocol
from physlab.experiments.library.pendulum import SIM_GRAVITY, pendulum_protocol
from physlab.experiments.protocol import (
    ExperimentProtocol,
    Measurement,
    ProcedureStep,
    SafetyClass,
    Variable,
    validate_protocol,
)
from physlab.experiments.registry import SkillRegistry
from physlab.experiments.results import RunResult, TrialContext
from physlab.experiments.runner import ExperimentRunner


def test_fit_recovers_a_noisy_power_law() -> None:
    generator = np.random.default_rng(0)
    xs = np.linspace(0.2, 1.2, 40)
    ys = 2.5 * xs**0.5 + generator.normal(0.0, 0.002, size=xs.size)
    frame = pd.DataFrame({"x": xs, "y": ys})
    fits = fit_models(frame, "x", "y")
    assert [item.name for item in fits] == sorted(
        (item.name for item in fits), key=lambda name: _aic(fits, name)
    )
    power = next(item for item in fits if item.name == "power")
    assert fits[0].name == "power"
    assert abs(power.params["a"] - 2.5) < 0.05
    assert abs(power.params["b"] - 0.5) < 0.05


def test_fit_custom_line_and_rejects_calls() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [4.0, 7.0, 10.0, 13.0]})
    fit = fit_custom(frame, "a * x + b", "x", "y", ["a", "b"], p0=[1.0, 1.0])
    assert abs(fit.params["a"] - 3.0) < 1e-6
    assert abs(fit.params["b"] - 1.0) < 1e-6
    compiled = compile_expression("sqrt(x ** 2)", [])
    assert compiled(4.0, {}) == pytest.approx(4.0)
    with pytest.raises(ConfigError):
        compile_expression("__import__('os')", [])
    with pytest.raises(ConfigError):
        compile_expression("eval(x)", [])


def test_runner_keeps_failed_trials(tmp_path: Path) -> None:
    registry = SkillRegistry()

    def boom(ctx: TrialContext) -> None:
        raise RuntimeError(f"sensor unplugged at {ctx.values['length_m']}")

    registry.register("boom", boom)
    protocol = _protocol(["boom"], [0.2, 0.4], repetitions=2)
    result = ExperimentRunner(EventLog(tmp_path, seed=0), skills=registry).run(protocol)
    assert len(result.trials) == 4
    assert all(not trial.success for trial in result.trials)
    assert "sensor unplugged" in result.trials[0].reason
    table = pd.read_parquet(result.directory / "trials.parquet")
    assert len(table) == 4
    assert not bool(table["success"].any())
    assert (result.directory / "protocol.json").is_file()
    assert (result.directory / "events.jsonl").is_file()
    assert (result.directory / "media").is_dir()


def test_unknown_skill_is_a_validation_error() -> None:
    protocol = _protocol(["not_a_skill"], [1.0], repetitions=1)
    issues = validate_protocol(protocol, ["boom"])
    assert "unknown skill not_a_skill" in issues


def test_seeded_shuffle_is_stable(tmp_path: Path) -> None:
    registry = _echo_registry()
    first = ExperimentRunner(EventLog(tmp_path / "a", seed=1), skills=registry).run(_grid(0))
    second = ExperimentRunner(EventLog(tmp_path / "b", seed=2), skills=registry).run(_grid(0))
    third = ExperimentRunner(EventLog(tmp_path / "c", seed=3), skills=registry).run(_grid(1))
    order = [
        (trial.values["left"], trial.values["right"], trial.repetition) for trial in first.trials
    ]
    again = [
        (trial.values["left"], trial.values["right"], trial.repetition) for trial in second.trials
    ]
    other = [
        (trial.values["left"], trial.values["right"], trial.repetition) for trial in third.trials
    ]
    assert order == again
    assert order != other
    assert len(order) == 8


@pytest.mark.slow
def test_pendulum_law(tmp_path: Path) -> None:
    result = ExperimentRunner(EventLog(tmp_path, seed=0)).run(pendulum_protocol(0))
    frame = _frame(result, "length_m", "period_s")
    power = next(item for item in fit_models(frame, "length_m", "period_s") if item.name == "power")
    gravity = fit_custom(
        frame,
        "6.283185307179586 * sqrt(x / g)",
        "length_m",
        "period_s",
        ["g"],
        p0=[9.8],
    )
    assert abs(power.params["b"] - 0.5) <= 0.05
    assert abs(gravity.params["g"] - SIM_GRAVITY) / SIM_GRAVITY <= 0.05


def test_incline_friction(tmp_path: Path) -> None:
    result = ExperimentRunner(EventLog(tmp_path, seed=0)).run(incline_protocol(0))
    rows = [
        {
            "angle_rad": np.deg2rad(trial.values["angle_deg"]),
            "accel": trial.measurements["accel_m_s2"],
        }
        for trial in result.trials
    ]
    fit = fit_custom(
        pd.DataFrame(rows),
        "9.81 * (sin(x) - mu * cos(x))",
        "angle_rad",
        "accel",
        ["mu"],
        p0=[0.05],
    )
    assert abs(fit.params["mu"] - SIM_FRICTION) <= 0.05


def test_thin_lens_focal_length(tmp_path: Path) -> None:
    result = ExperimentRunner(EventLog(tmp_path, seed=0)).run(lens_protocol(0))
    frame = _frame(result, "object_m", "image_m")
    fit = fit_custom(frame, "x * f / (x - f)", "object_m", "image_m", ["f"], p0=[0.04])
    assert abs(fit.params["f"] - TRUE_FOCAL_M) / TRUE_FOCAL_M <= 0.03
    path = report(result, [fit], "object_m", "image_m")
    text = path.read_text(encoding="utf-8")
    assert "E3_thin_lens" in text
    assert "media/" in text
    assert any((result.directory / "media").glob("*.png"))


def _aic(fits: list[ModelFit], name: str) -> float:
    for item in fits:
        if item.name == name:
            return item.aic
    raise AssertionError(name)


def _protocol(skills: list[str], lengths: list[float], repetitions: int) -> ExperimentProtocol:
    return ExperimentProtocol(
        id="boom",
        question="Does the runner keep failures?",
        hypothesis="Failed trials stay in the table.",
        apparatus="none",
        independent=[Variable(name="length_m", unit="m", values=lengths)],
        dependent=[Measurement(name="period_s", unit="s", method="none")],
        procedure=[ProcedureStep(skill=name) for name in skills],
        repetitions=repetitions,
        seed=0,
        safety_class=SafetyClass.SIM_ONLY,
    )


def _echo_registry() -> SkillRegistry:
    registry = SkillRegistry()

    def echo(ctx: TrialContext) -> None:
        ctx.records["sum"] = ctx.values["left"] + ctx.values["right"]

    registry.register("echo", echo)
    return registry


def _grid(seed: int) -> ExperimentProtocol:
    return ExperimentProtocol(
        id="grid",
        question="Is the factorial complete?",
        hypothesis="Every cell runs once per repetition.",
        apparatus="none",
        independent=[
            Variable(name="left", unit="1", values=[1.0, 2.0]),
            Variable(name="right", unit="1", values=[3.0, 4.0]),
        ],
        dependent=[Measurement(name="sum", unit="1", method="add")],
        procedure=[ProcedureStep(skill="echo")],
        repetitions=2,
        seed=seed,
        safety_class=SafetyClass.SIM_ONLY,
    )


def _frame(result: RunResult, x: str, y: str) -> pd.DataFrame:
    rows = [
        {x: trial.values[x], y: trial.measurements[y]} for trial in result.trials if trial.success
    ]
    return pd.DataFrame(rows)
