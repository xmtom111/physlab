"""Serial-dilute food dye and fit its molar absorptivity.

The slope of absorbance versus concentration, divided by the 1 cm path in
the reagent file, should match the allowlisted epsilon within 3%.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from physlab.chem.safety import ChemSafetyPolicy
from physlab.chem.workflows import DYE_WAVELENGTH_NM, dye_calibration_protocol
from physlab.core.events import EventLog
from physlab.core.logging import configure_logging, get_logger
from physlab.experiments.analysis import fit_models
from physlab.experiments.runner import ExperimentRunner

log = get_logger(__name__)


def main() -> None:
    """Fit the dye curve and exit non-zero when epsilon is outside 3%.

    Raises:
        SystemExit: The recovered molar absorptivity misses the allowlist.
    """
    configure_logging("INFO", False)
    policy = ChemSafetyPolicy.load()
    result = ExperimentRunner(EventLog(Path("runs"), seed=0)).run_chem(
        dye_calibration_protocol(),
        question="How does food-dye absorbance depend on concentration?",
        hypothesis="Beer-Lambert: absorbance is linear in concentration.",
        seed=0,
    )
    frame = pd.DataFrame(
        [{**trial.values, **trial.measurements} for trial in result.trials if trial.success]
    )
    fit = next(
        item for item in fit_models(frame, "concentration_m", "absorbance") if item.name == "linear"
    )
    epsilon = fit.params["b"] / policy.allowlist.path_cm
    expected = policy.allowlist.reagents["food_dye"].epsilon_l_mol_cm[int(DYE_WAVELENGTH_NM)]
    ok = abs(epsilon - expected) / expected <= 0.03
    log.info(
        "dye_calibration", epsilon=epsilon, expected=expected, ok=ok, path=str(result.directory)
    )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
