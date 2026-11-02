"""Write one experiment run under ``runs/<run_id>/``.

The directory holds ``protocol.json``, ``trials.parquet``, the event log's
``events.jsonl``, and a ``media`` folder for plots.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from physlab.experiments.protocol import ExperimentProtocol
from physlab.experiments.results import TrialRecord


def write_protocol(directory: Path, protocol: ExperimentProtocol) -> Path:
    """Write the protocol JSON.

    Args:
        directory: Run directory.
        protocol: Protocol that was executed.

    Returns:
        Path of ``protocol.json``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "media").mkdir(exist_ok=True)
    path = directory / "protocol.json"
    path.write_text(json.dumps(protocol.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path


def write_trials(directory: Path, trials: list[TrialRecord]) -> Path:
    """Write the trial table.

    Args:
        directory: Run directory.
        trials: Every trial, including failures.

    Returns:
        Path of ``trials.parquet``.
    """
    rows = [_row(trial) for trial in trials]
    frame = pd.DataFrame(rows or [{"order": 0}])
    path = directory / "trials.parquet"
    frame.to_parquet(path, index=False)
    return path


def _row(trial: TrialRecord) -> dict[str, object]:
    row: dict[str, object] = {
        "order": trial.order,
        "repetition": trial.repetition,
        "success": trial.success,
        "reason": trial.reason,
    }
    row.update(trial.values)
    row.update(trial.measurements)
    return row
