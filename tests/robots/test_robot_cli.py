"""Robot and asset commands."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from physlab.cli import app

_runner = CliRunner()


def test_robots_list_table() -> None:
    result = _runner.invoke(app, ["robots", "list"])
    assert result.exit_code == 0
    assert "id" in result.stdout
    assert "ufactory_xarm7" in result.stdout
    assert "generic_diffdrive" in result.stdout
    assert "gantry_xyz" in result.stdout


def test_robots_check_requires_a_target() -> None:
    result = _runner.invoke(app, ["robots", "check"])
    assert result.exit_code == 2
    missing = _runner.invoke(app, ["robots", "check", "not_a_robot"])
    assert missing.exit_code == 1
    assert "FAIL" in missing.stdout


@pytest.mark.assets
def test_robots_check_all() -> None:
    result = _runner.invoke(app, ["robots", "check", "--all"])
    assert result.exit_code == 0, result.stdout
    assert "FAIL" not in result.stdout
    assert "OK ufactory_xarm7" in result.stdout
    assert "OK hello_robot_stretch_3" in result.stdout
    assert "OK generic_diffdrive" in result.stdout
    assert "OK gantry_xyz" in result.stdout
