"""Configuration loading and environment overrides."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from physlab.core.config import GlBackend, load_config
from physlab.core.errors import ConfigError
from physlab.core.types import Mode

_YAML = """\
mode: sim
run_dir: runs
assets_dir: assets
configs_dir: configs
seed: 0
sim:
  timestep_s: 0.002
  render_width: 640
  render_height: 480
  gl_backend: egl
"""


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("PHYSLAB_"):
            monkeypatch.delenv(key, raising=False)


def test_default_config_and_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = load_config()
    assert loaded.mode is Mode.SIM
    assert loaded.seed == 0
    assert loaded.sim.timestep_s == pytest.approx(0.002)
    assert loaded.sim.gl_backend is GlBackend.EGL
    assert loaded.run_dir.is_absolute()
    assert loaded.run_dir.name == "runs"
    assert loaded.assets_dir.name == "assets"
    assert loaded.configs_dir.name == "configs"
    monkeypatch.setenv("PHYSLAB_MODE", "real")
    monkeypatch.setenv("PHYSLAB_SEED", "4")
    monkeypatch.setenv("PHYSLAB_SIM__TIMESTEP_S", "0.001")
    monkeypatch.setenv("PHYSLAB_SIM__RENDER_WIDTH", "320")
    monkeypatch.setenv("PHYSLAB_SIM__GL_BACKEND", "osmesa")
    monkeypatch.setenv("PHYSLAB_RUN_DIR", "/tmp/physlab-runs")
    overridden = load_config()
    assert overridden.mode is Mode.REAL
    assert overridden.seed == 4
    assert overridden.sim.timestep_s == pytest.approx(0.001)
    assert overridden.sim.render_width == 320
    assert overridden.sim.render_height == 480
    assert overridden.sim.gl_backend is GlBackend.OSMESA
    assert overridden.run_dir == Path("/tmp/physlab-runs")


def test_explicit_path_resolves_against_config_root(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "lab.yaml"
    config_path.write_text(_YAML, encoding="utf-8")
    loaded = load_config(config_path)
    assert loaded.run_dir == (tmp_path / "runs").resolve()
    assert loaded.assets_dir == (tmp_path / "assets").resolve()


def test_discovery_away_from_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    loaded = load_config()
    assert loaded.mode is Mode.SIM
    assert loaded.sim.timestep_s == pytest.approx(0.002)


def test_invalid_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(ConfigError):
        load_config(missing)
    broken = tmp_path / "broken.yaml"
    broken.write_text("mode: [\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(broken)
    listing = tmp_path / "list.yaml"
    listing.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(listing)
    monkeypatch.setenv("PHYSLAB_MODE", "banana")
    with pytest.raises(ConfigError):
        load_config()
    monkeypatch.setenv("PHYSLAB_MODE__NESTED", "1")
    with pytest.raises(ConfigError):
        load_config()
