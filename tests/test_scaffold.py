"""Repository layout for modules that later tasks fill in."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest

_PACKAGES = (
    "physlab",
    "physlab.core",
    "physlab.hal",
    "physlab.robots",
    "physlab.sim",
    "physlab.motion",
    "physlab.perception",
    "physlab.fabrication",
    "physlab.fabrication.parts",
    "physlab.optics",
    "physlab.assembly",
    "physlab.controllers",
    "physlab.experiments",
    "physlab.experiments.library",
    "physlab.chem",
    "physlab.safety",
    "physlab.real",
    "physlab.mcp_server",
    "physlab.cli",
)


@pytest.mark.parametrize("name", _PACKAGES)
def test_package_imports(name: str) -> None:
    importlib.import_module(name)


def test_fetch_assets_is_t2_stub() -> None:
    path = Path(__file__).resolve().parents[1] / "scripts" / "fetch_assets.py"
    spec = importlib.util.spec_from_file_location("fetch_assets_stub", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(NotImplementedError, match="Implemented in T2"):
        module.fetch_assets()
