"""Run the pendulum length sweep and write a Markdown report.

The power-law exponent should be 0.5 within 0.05, and the gravity implied
by ``T = 2 pi sqrt(L / g)`` should be within 5% of the simulator gravity.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from physlab.core.events import EventLog
from physlab.core.logging import configure_logging, get_logger
from physlab.experiments.analysis import fit_custom, fit_models, report
from physlab.experiments.library.pendulum import SIM_GRAVITY, pendulum_protocol
from physlab.experiments.runner import ExperimentRunner

log = get_logger(__name__)


def main() -> None:
    """Fit the period law and exit non-zero when the bars are missed.

    Raises:
        SystemExit: The exponent or the recovered gravity is outside the bar.
    """
    configure_logging("INFO", False)
    result = ExperimentRunner(EventLog(Path("runs"), seed=0)).run(pendulum_protocol(0))
    frame = pd.DataFrame(
        [
            {"length_m": trial.values["length_m"], "period_s": trial.measurements["period_s"]}
            for trial in result.trials
            if trial.success
        ]
    )
    fits = fit_models(frame, "length_m", "period_s")
    gravity = fit_custom(
        frame,
        "6.283185307179586 * sqrt(x / g)",
        "length_m",
        "period_s",
        ["g"],
        p0=[SIM_GRAVITY],
    )
    path = report(result, [*fits, gravity], "length_m", "period_s")
    power = next(item for item in fits if item.name == "power")
    exponent_ok = abs(power.params["b"] - 0.5) <= 0.05
    gravity_ok = abs(gravity.params["g"] - SIM_GRAVITY) / SIM_GRAVITY <= 0.05
    log.info(
        "pendulum_report",
        path=str(path),
        exponent=power.params["b"],
        gravity=gravity.params["g"],
        exponent_ok=exponent_ok,
        gravity_ok=gravity_ok,
    )
    if not exponent_ok or not gravity_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
