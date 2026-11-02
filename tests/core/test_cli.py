"""Command line entry points."""

from __future__ import annotations

import pytest
import yaml
from typer.testing import CliRunner

from physlab import __version__
from physlab.cli import app


def test_version_is_0_1_0() -> None:
    assert __version__ == "0.1.0"
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"


def test_config_show_prints_yaml() -> None:
    result = CliRunner().invoke(app, ["config", "show"])
    assert result.exit_code == 0
    loaded = yaml.safe_load(result.stdout)
    assert loaded["mode"] == "sim"
    assert loaded["seed"] == 0
    assert loaded["sim"]["timestep_s"] == pytest.approx(0.002)
    assert loaded["sim"]["gl_backend"] == "egl"
