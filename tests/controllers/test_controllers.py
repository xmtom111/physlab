"""Static checks, the process sandbox, and the focus PID."""

from __future__ import annotations

import importlib.util
import stat
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from physlab.controllers.api import ActuatorChannel, DeviceManifest, SensorChannel
from physlab.controllers.codecheck import check_source
from physlab.controllers.firmware import compile_arduino, upload_arduino
from physlab.controllers.runner import Phase, run_controller
from physlab.core.errors import DependencyMissing, HardwareError
from physlab.sim.virtual_devices import StepperMotor, Thermometer

_HEADER = "from physlab.controllers.api import Controller\n"


def _load(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "examples" / "scenarios" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_codecheck_rejects_each_forbidden_construct() -> None:
    samples = {
        "os": "import os\n",
        "sys": "import sys\n",
        "subprocess": "import subprocess\n",
        "socket": "import socket\n",
        "threading": "import threading\n",
        "time": "import time\n",
        "open": _HEADER + "open('x')\n",
        "exec": _HEADER + "exec('1')\n",
        "eval": _HEADER + "eval('1')\n",
        "compile": _HEADER + "compile('1', '<c>', 'exec')\n",
        "__import__": _HEADER + "__import__('os')\n",
        "globals": _HEADER + "globals()\n",
        "locals": _HEADER + "locals()\n",
        "__class__": (
            _HEADER + "class A:\n    def step(self, ctx, obs):\n        return self.__class__\n"
        ),
        "getattr": _HEADER + "getattr(object, '__dict__')\n",
    }
    for name, source in samples.items():
        result = check_source(source)
        assert not result.ok, name
        assert any(item.name == name or name in item.message for item in result.violations)
        assert result.violations[0].line >= 1


def test_infinite_loop_times_out(tmp_path: Path) -> None:
    source = _HEADER + (
        "class Spin(Controller):\n"
        "    def step(self, ctx, obs):\n"
        "        while True:\n"
        "            ctx.log('n', 1)\n"
        "        return {}\n"
    )
    report = run_controller(
        source,
        _box_manifest(),
        _Box(),
        5.0,
        {},
        0,
        output_dir=tmp_path,
        step_timeout_s=0.4,
        wall_timeout_s=5.0,
        control_hz=20,
    )
    assert report.phase_failed is Phase.TIMEOUT
    assert not report.ok
    assert len(report.summary_text) <= 2000


def test_step_exception_reports_the_source_line(tmp_path: Path) -> None:
    source = (
        _HEADER
        + "class Boom(Controller):\n"
        + "    def step(self, ctx, obs):\n"
        + "        raise RuntimeError('boom')\n"
    )
    raise_line = source.splitlines().index("        raise RuntimeError('boom')") + 1
    report = run_controller(
        source, _box_manifest(), _Box(), 0.2, {}, 0, output_dir=tmp_path, control_hz=20
    )
    assert report.phase_failed is Phase.STEP
    assert f"line {raise_line}" in report.traceback


def test_pid_focus_reaches_focus(tmp_path: Path) -> None:
    source = _HEADER + (
        "class Pid(Controller):\n"
        "    def step(self, ctx, obs):\n"
        "        return {'stage_rate': -ctx.params['kp'] * obs['defocus_m']}\n"
    )
    scenario = _load("focus_stage").FocusStage()
    report = run_controller(
        source,
        scenario.manifest,
        scenario,
        1.2,
        {"kp": 1.2e6},
        0,
        output_dir=tmp_path,
        control_hz=100,
    )
    assert report.ok
    frame = pd.read_parquet(report.metrics_path)
    assert abs(float(frame["defocus_m"].iloc[-1])) < 50e-6
    assert report.plots
    assert all(path.stat().st_size > 0 for path in report.plots)


def test_clipping_is_reported(tmp_path: Path) -> None:
    source = _HEADER + (
        "class High(Controller):\n    def step(self, ctx, obs):\n        return {'cmd': 5.0}\n"
    )
    plant = _Box()
    report = run_controller(
        source, plant.manifest, plant, 0.1, {}, 0, output_dir=tmp_path, control_hz=20
    )
    assert report.ok
    assert report.clipping_events
    event = report.clipping_events[0]
    assert event.commanded == 5.0
    assert event.applied == 1.0
    assert plant.value == pytest.approx(1.0)


def test_stepper_noise_repeats_for_the_same_seed() -> None:
    first = StepperMotor(200, 1000, seed=3, noise_std=0.25)
    second = StepperMotor(200, 1000, seed=3, noise_std=0.25)
    assert first.read_steps() == second.read_steps()
    other = StepperMotor(200, 1000, seed=4, noise_std=0.25)
    assert first.read_steps() != other.read_steps()


def test_thermometer_relaxes_toward_the_heater_target() -> None:
    sensor = Thermometer(300.0, tau_s=1.0, gain_k_per_watt=10.0, seed=0)
    sensor.step(1.0, 0.1)
    assert sensor.temperature_k == pytest.approx(301.0)
    assert sensor.read() == pytest.approx(301.0)


def test_pendulum_falls_when_released() -> None:
    plant = _load("pendulum_release").PendulumRelease()
    plant.reset(0)
    plant.apply({"servo_rad": 0.4, "hold": 0.0}, 0.05)
    assert plant.angle_rad < 0.4


def test_line_sensors_see_the_near_side_of_the_line() -> None:
    plant = _load("line_follow").LineFollow()
    plant.reset(0)
    reading = plant.measure()
    assert reading["photo_left"] > reading["photo_right"]
    assert reading["photo_left"] == pytest.approx(1.0)


def test_compile_arduino_requires_the_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(DependencyMissing):
        compile_arduino(tmp_path, "arduino:avr:uno")


def test_upload_arduino_uses_a_fake_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = tmp_path / "arduino-cli"
    binary.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", str(tmp_path))
    result = upload_arduino(tmp_path, "arduino:avr:uno", "/dev/ttyUSB0")
    assert result.returncode == 0
    assert "upload" in result.stdout
    binary.write_text("#!/bin/sh\necho nope >&2\nexit 3\n", encoding="utf-8")
    with pytest.raises(HardwareError):
        compile_arduino(tmp_path, "arduino:avr:uno")


def _box_manifest() -> DeviceManifest:
    return DeviceManifest(
        sensors={"x": SensorChannel(unit="m")},
        actuators={"cmd": ActuatorChannel(unit="1", min=-1.0, max=1.0, rate_limit=0.0)},
    )


class _Box:
    def __init__(self) -> None:
        self.manifest = _box_manifest()
        self.value = 0.0

    def reset(self, seed: int) -> None:
        del seed
        self.value = 0.0

    def measure(self) -> dict[str, float]:
        return {"x": self.value}

    def apply(self, action: dict[str, float], dt_s: float) -> None:
        del dt_s
        self.value = action["cmd"]


def test_math_import_is_allowed() -> None:
    source = "import math\nfrom physlab.controllers.api import Controller\n"
    assert check_source(source).ok
