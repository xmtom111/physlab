"""Run a focus controller that drives the wrong way, then the corrected one.

The buggy law uses the sign that increases defocus. The fixed law is a
proportional step rate on ``defocus_m``. Both summaries are printed so a
later agent can compare them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from physlab.controllers.runner import run_controller
from physlab.core.logging import configure_logging

_BUGGY = """
from physlab.controllers.api import Controller

class BuggyFocus(Controller):
    def step(self, ctx, obs):
        # The positive sign drives the stage further from the focal plane.
        return {"stage_rate": ctx.params["kp"] * obs["defocus_m"]}
"""

_FIXED = """
from physlab.controllers.api import Controller

class FixedFocus(Controller):
    def step(self, ctx, obs):
        return {"stage_rate": -ctx.params["kp"] * obs["defocus_m"]}
"""


def main() -> None:
    """Print both summaries and exit non-zero if the fix misses focus.

    Raises:
        SystemExit: The corrected controller is still outside 50 µm.
    """
    configure_logging("INFO", False)
    scenario = _focus_stage()()
    params = {"kp": 1.2e6}
    buggy = run_controller(_BUGGY, scenario.manifest, scenario, 0.8, params, 0, control_hz=100)
    print(buggy.summary_text)
    fixed = run_controller(_FIXED, scenario.manifest, scenario, 1.2, params, 0, control_hz=100)
    print(fixed.summary_text)
    frame = pd.read_parquet(fixed.metrics_path)
    final = float(frame["defocus_m"].iloc[-1])
    if not fixed.ok or abs(final) > 50e-6:
        raise SystemExit(1)


def _focus_stage() -> type:
    path = Path(__file__).resolve().parent / "scenarios" / "focus_stage.py"
    spec = importlib.util.spec_from_file_location("focus_stage", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FocusStage


if __name__ == "__main__":
    main()
