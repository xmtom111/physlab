"""Run a checked controller against a scenario the parent process owns.

Actions are clipped to the manifest before they touch the plant. Clipping is
reported rather than hidden. Metrics go to parquet and one PNG per channel.
``summary_text`` is capped at 2000 characters so an agent can read it back.
"""

from __future__ import annotations

import multiprocessing
import time
from collections.abc import Mapping
from enum import StrEnum
from multiprocessing.connection import Connection
from pathlib import Path
from typing import ClassVar, Protocol

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from physlab.controllers.api import ActuatorChannel, DeviceManifest
from physlab.controllers.codecheck import CodeCheckResult, check_source
from physlab.controllers.sandbox import child_main
from physlab.core.logging import get_logger

_LOG = get_logger(__name__)
_SUMMARY_LIMIT = 2000


class Phase(StrEnum):
    """Where a controller run stopped, when it did not finish cleanly."""

    CHECK = "CHECK"
    IMPORT = "IMPORT"
    SETUP = "SETUP"
    STEP = "STEP"
    TEARDOWN = "TEARDOWN"
    TIMEOUT = "TIMEOUT"


class AssertionFailure(BaseModel):
    """One failed :meth:`ControllerContext.assert_that`.

    Attributes:
        time_s: Controller time when the assertion failed.
        message: Text the controller supplied.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    time_s: float
    message: str


class ClippingEvent(BaseModel):
    """One actuator command the runner changed.

    Attributes:
        time_s: Controller time of the tick.
        name: Actuator name.
        commanded: Value the controller returned.
        applied: Value forwarded to the scenario.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    time_s: float
    name: str
    commanded: float
    applied: float


class RunReport(BaseModel):
    """What the parent kept after a controller run.

    Attributes:
        ok: True when the run finished, no assertion failed, and no phase failed.
        phase_failed: Failing phase, or ``None`` when the loop completed.
        traceback: Child traceback, empty when there was none.
        stdout_tail: Tail of the child's stdout and stderr.
        metrics_path: Parquet time series.
        assertion_failures: Recorded assertion failures.
        clipping_events: Commands changed by range or rate limits.
        plots: One PNG per sensor and actuator.
        summary_text: At most 2000 characters, written for an LLM.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    ok: bool
    phase_failed: Phase | None
    traceback: str = ""
    stdout_tail: str = ""
    metrics_path: Path
    assertion_failures: list[AssertionFailure] = Field(default_factory=list)
    clipping_events: list[ClippingEvent] = Field(default_factory=list)
    plots: list[Path] = Field(default_factory=list)
    summary_text: str = ""


class Scenario(Protocol):
    """Plant the parent steps. It is never sent to the controller process."""

    manifest: DeviceManifest

    def reset(self, seed: int) -> None:
        """Restore the plant. ``seed`` seeds device noise."""

    def measure(self) -> Mapping[str, float]:
        """Return true sensor values before manifest noise."""

    def apply(self, action: Mapping[str, float], dt_s: float) -> None:
        """Apply already clipped actuator commands."""


def run_controller(
    source_code: str,
    manifest: DeviceManifest,
    scenario: Scenario,
    duration_s: float,
    params: Mapping[str, float],
    seed: int,
    *,
    control_hz: float = 100.0,
    output_dir: Path | None = None,
    step_timeout_s: float = 2.0,
    wall_timeout_s: float | None = None,
) -> RunReport:
    """Check ``source_code``, spawn it, and step ``scenario`` at ``control_hz``.

    Args:
        source_code: Controller program.
        manifest: Channels used for noise, clipping, and the report.
        scenario: Plant stepped in this process.
        duration_s: Simulated time to run, seconds.
        params: Numeric parameters copied into the controller context.
        seed: Seed for sensor noise and the scenario.
        control_hz: Control rate. The default is 100 Hz.
        output_dir: Directory for parquet and plots. A ``runs/controllers``
            folder is created when this is omitted.
        step_timeout_s: Wall time allowed for one ``step`` call.
        wall_timeout_s: Wall time for the whole exchange. The default is
            ``duration_s`` plus 15 seconds.

    Returns:
        Report. ``phase_failed`` is ``CHECK`` when the static check fails,
        and ``TIMEOUT`` when a step or the process misses its wall budget.
    """
    directory = _prepare(output_dir)
    checked = check_source(source_code)
    if not checked.ok:
        return _checked_failure(directory, checked)
    scenario.reset(seed)
    wall = duration_s + 15.0 if wall_timeout_s is None else wall_timeout_s
    return _exchange(
        source_code,
        manifest,
        scenario,
        duration_s,
        dict(params),
        seed,
        control_hz,
        directory,
        step_timeout_s,
        wall,
    )


def _prepare(output_dir: Path | None) -> Path:
    directory = output_dir or Path("runs") / "controllers" / time.strftime("%Y%m%d-%H%M%S")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _checked_failure(directory: Path, checked: CodeCheckResult) -> RunReport:
    lines = [f"{item.line}: {item.message}" for item in checked.violations]
    summary = _clip_text("CHECK failed\n" + "\n".join(lines))
    path = _write_metrics(directory, [])
    return _report(directory, path, [], [], Phase.CHECK, summary, traceback=summary)


def _exchange(
    source: str,
    manifest: DeviceManifest,
    scenario: Scenario,
    duration_s: float,
    params: dict[str, float],
    seed: int,
    control_hz: float,
    directory: Path,
    step_timeout_s: float,
    wall_s: float,
) -> RunReport:
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe(duplex=True)
    stdout_path = directory / "stdout.txt"
    cpu_s = max(2, int(wall_s) + 1)
    proc = ctx.Process(
        target=child_main,
        args=(source, child, params, 1.0 / control_hz, str(stdout_path), cpu_s),
    )
    proc.start()
    child.close()
    try:
        return _loop(
            parent,
            proc,
            manifest,
            scenario,
            duration_s,
            seed,
            control_hz,
            directory,
            step_timeout_s,
            wall_s,
        )
    finally:
        _stop(proc)
        parent.close()


def _loop(
    parent: Connection,
    proc: multiprocessing.context.SpawnProcess,
    manifest: DeviceManifest,
    scenario: Scenario,
    duration_s: float,
    seed: int,
    control_hz: float,
    directory: Path,
    step_timeout_s: float,
    wall_s: float,
) -> RunReport:
    deadline = time.monotonic() + wall_s
    ready, phase, trace, assertions = _await(parent, deadline, step_timeout_s)
    if ready is None:
        return _failed(directory, phase or Phase.TIMEOUT, trace, assertions, [], [])
    rows: list[dict[str, float]] = []
    clips: list[ClippingEvent] = []
    previous = {name: 0.0 for name in manifest.actuators}
    rng = np.random.default_rng(seed)
    dt_s = 1.0 / control_hz
    failure = _ticks(
        parent,
        manifest,
        scenario,
        duration_s,
        dt_s,
        deadline,
        step_timeout_s,
        rng,
        previous,
        rows,
        clips,
        assertions,
    )
    if failure is not None:
        return _failed(directory, failure[0], failure[1], assertions, clips, rows)
    return _success(directory, rows, clips, assertions)


def _ticks(
    parent: Connection,
    manifest: DeviceManifest,
    scenario: Scenario,
    duration_s: float,
    dt_s: float,
    deadline: float,
    step_timeout_s: float,
    rng: np.random.Generator,
    previous: dict[str, float],
    rows: list[dict[str, float]],
    clips: list[ClippingEvent],
    assertions: list[AssertionFailure],
) -> tuple[Phase, str] | None:
    steps = max(1, round(duration_s / dt_s))
    for index in range(steps):
        now = index * dt_s
        obs = _noisy(manifest, scenario.measure(), rng)
        parent.send({"cmd": "obs", "time_s": now, "obs": obs})
        message, phase, trace, extra = _await(parent, deadline, step_timeout_s)
        assertions.extend(extra)
        if message is None or message.get("cmd") != "action":
            return phase or Phase.TIMEOUT, trace
        applied, events = _clip(manifest, _as_floats(message.get("action")), previous, dt_s, now)
        clips.extend(events)
        scenario.apply(applied, dt_s)
        rows.append(_row(now, obs, applied, _as_floats(message.get("logs"))))
    return _finish(parent, deadline, step_timeout_s, assertions)


def _finish(
    parent: Connection,
    deadline: float,
    step_timeout_s: float,
    assertions: list[AssertionFailure],
) -> tuple[Phase, str] | None:
    parent.send({"cmd": "stop"})
    message, phase, trace, extra = _await(parent, deadline, step_timeout_s)
    assertions.extend(extra)
    if message is None or message.get("cmd") != "done":
        return phase or Phase.TEARDOWN, trace
    return None


def _await(
    parent: Connection, deadline: float, step_timeout_s: float
) -> tuple[dict[str, object] | None, Phase | None, str, list[AssertionFailure]]:
    remaining = deadline - time.monotonic()
    timeout = min(step_timeout_s, max(0.0, remaining))
    if timeout <= 0.0 or not parent.poll(timeout):
        return None, Phase.TIMEOUT, "wall timeout waiting for the controller", []
    message = parent.recv()
    if not isinstance(message, dict):
        return None, Phase.STEP, "controller sent a non-object message", []
    payload = {str(key): value for key, value in message.items()}
    if payload.get("cmd") == "error":
        phase = Phase(str(payload.get("phase", "STEP")))
        trace = str(payload.get("traceback", ""))
        return None, phase, trace, []
    assertions: list[AssertionFailure] = []
    _collect_assertions(assertions, payload)
    return payload, None, "", assertions


def _noisy(
    manifest: DeviceManifest, true: Mapping[str, float], rng: np.random.Generator
) -> dict[str, float]:
    observed: dict[str, float] = {}
    for name, spec in manifest.sensors.items():
        value = float(true.get(name, 0.0))
        if spec.noise_std > 0.0:
            value += float(rng.normal(0.0, spec.noise_std))
        low, high = spec.range
        observed[name] = float(min(max(value, low), high))
    return observed


def _clip(
    manifest: DeviceManifest,
    commanded: dict[str, float],
    previous: dict[str, float],
    dt_s: float,
    time_s: float,
) -> tuple[dict[str, float], list[ClippingEvent]]:
    applied: dict[str, float] = {}
    events: list[ClippingEvent] = []
    for name, spec in manifest.actuators.items():
        raw = commanded.get(name, previous[name])
        value = _limit(raw, previous[name], spec, dt_s)
        previous[name] = value
        applied[name] = value
        if value != raw:
            events.append(ClippingEvent(time_s=time_s, name=name, commanded=raw, applied=value))
    return applied, events


def _limit(raw: float, previous: float, spec: ActuatorChannel, dt_s: float) -> float:
    value = min(max(raw, spec.min), spec.max)
    if spec.rate_limit > 0.0:
        window = spec.rate_limit * dt_s
        value = min(max(value, previous - window), previous + window)
    return min(max(value, spec.min), spec.max)


def _row(
    time_s: float,
    obs: Mapping[str, float],
    action: Mapping[str, float],
    logs: Mapping[str, float],
) -> dict[str, float]:
    row = {"time_s": time_s, **obs, **action}
    for key, value in logs.items():
        row[key if key not in row else f"log_{key}"] = value
    return row


def _as_floats(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {str(key): float(item) for key, item in value.items()}


def _collect_assertions(sink: list[AssertionFailure], message: dict[str, object]) -> None:
    raw = message.get("assertions", [])
    if not isinstance(raw, list):
        return
    for item in raw:
        if isinstance(item, tuple) and len(item) == 2:
            sink.append(AssertionFailure(time_s=float(item[0]), message=str(item[1])))


def _failed(
    directory: Path,
    phase: Phase,
    trace: str,
    assertions: list[AssertionFailure],
    clips: list[ClippingEvent],
    rows: list[dict[str, float]],
) -> RunReport:
    path = _write_metrics(directory, rows)
    plots = _plots(directory, rows)
    summary = _summary(phase, trace, assertions, clips, rows)
    return _report(directory, path, assertions, clips, phase, summary, traceback=trace, plots=plots)


def _success(
    directory: Path,
    rows: list[dict[str, float]],
    clips: list[ClippingEvent],
    assertions: list[AssertionFailure],
) -> RunReport:
    path = _write_metrics(directory, rows)
    plots = _plots(directory, rows)
    summary = _summary(None, "", assertions, clips, rows)
    return _report(
        directory, path, assertions, clips, None, summary, plots=plots, ok=not assertions
    )


def _report(
    directory: Path,
    metrics_path: Path,
    assertions: list[AssertionFailure],
    clips: list[ClippingEvent],
    phase: Phase | None,
    summary: str,
    *,
    traceback: str = "",
    plots: list[Path] | None = None,
    ok: bool | None = None,
) -> RunReport:
    text = _clip_text(summary)
    (directory / "summary.txt").write_text(text, encoding="utf-8")
    logged = ok if ok is not None else phase is None
    _LOG.info("controller_run", ok=logged, phase=None if phase is None else phase.value)
    return RunReport(
        ok=phase is None if ok is None else ok,
        phase_failed=phase,
        traceback=traceback,
        stdout_tail=_tail(directory / "stdout.txt"),
        metrics_path=metrics_path,
        assertion_failures=assertions,
        clipping_events=clips,
        plots=plots or [],
        summary_text=text,
    )


def _summary(
    phase: Phase | None,
    trace: str,
    assertions: list[AssertionFailure],
    clips: list[ClippingEvent],
    rows: list[dict[str, float]],
) -> str:
    last = rows[-1] if rows else {}
    readings = ", ".join(f"{key}={value:.6g}" for key, value in last.items())
    lines = [
        f"phase={phase.value if phase else 'ok'}",
        f"ticks={len(rows)} clipping={len(clips)} assertions={len(assertions)}",
        f"last: {readings}" if readings else "last: none",
    ]
    lines.extend(f"assert t={item.time_s:.3f}: {item.message}" for item in assertions[:8])
    if clips:
        sample = clips[0]
        lines.append(f"clip {sample.name}: {sample.commanded:.6g} -> {sample.applied:.6g}")
    if trace:
        lines.append(trace[-800:])
    return "\n".join(lines)


def _write_metrics(directory: Path, rows: list[dict[str, float]]) -> Path:
    frame = pd.DataFrame(rows or [{"time_s": 0.0}])
    path = directory / "metrics.parquet"
    frame.to_parquet(path, index=False)
    return path


def _plots(directory: Path, rows: list[dict[str, float]]) -> list[Path]:
    if not rows:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(rows)
    written: list[Path] = []
    for column in frame.columns:
        if column == "time_s":
            continue
        figure, axis = plt.subplots()
        axis.plot(frame["time_s"], frame[column])
        axis.set_xlabel("time_s")
        axis.set_ylabel(column)
        path = directory / f"{column}.png"
        figure.savefig(path)
        plt.close(figure)
        written.append(path)
    return written


def _stop(proc: multiprocessing.context.SpawnProcess) -> None:
    if not proc.is_alive():
        return
    proc.terminate()
    proc.join(1)
    if proc.is_alive():
        proc.kill()
        proc.join(1)


def _tail(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-2000:]


def _clip_text(text: str) -> str:
    if len(text) <= _SUMMARY_LIMIT:
        return text
    return text[: _SUMMARY_LIMIT - 3] + "..."
