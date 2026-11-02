"""Titrate acetic acid with sodium hydroxide and find the equivalence point.

The steepest part of the pH curve should fall within 2% of the theoretical
titrant volume.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from physlab.chem.workflows import EQUIVALENCE_UL, titration_protocol
from physlab.core.events import EventLog
from physlab.core.logging import configure_logging, get_logger
from physlab.experiments.runner import ExperimentRunner

log = get_logger(__name__)


def main() -> None:
    """Locate the equivalence volume and exit non-zero when it misses 2%.

    Raises:
        SystemExit: The steepest segment is more than 2% from theory.
    """
    configure_logging("INFO", False)
    result = ExperimentRunner(EventLog(Path("runs"), seed=0)).run_chem(
        titration_protocol(),
        question="Where is the equivalence point of acetic acid and NaOH?",
        hypothesis="The steepest pH rise is at n_acid / C_base.",
        seed=0,
        analyte="acetic_acid",
    )
    volumes = np.array([trial.values["titrant_ul"] for trial in result.trials])
    ph = np.array([trial.measurements["ph"] for trial in result.trials])
    found = _equivalence_ul(volumes, ph)
    error = abs(found - EQUIVALENCE_UL) / EQUIVALENCE_UL
    ok = error <= 0.02
    log.info("titration", equivalence_ul=found, theory_ul=EQUIVALENCE_UL, error=error, ok=ok)
    if not ok:
        raise SystemExit(1)


def _equivalence_ul(volumes: np.ndarray, ph: np.ndarray) -> float:
    slopes = np.diff(ph) / np.diff(volumes)
    index = int(np.argmax(slopes))
    return float(0.5 * (volumes[index] + volumes[index + 1]))


if __name__ == "__main__":
    main()
