"""Catalog loading and MuJoCo name checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from physlab.core.errors import ConfigError
from physlab.robots.catalog import RobotCatalog
from physlab.robots.compose import compose_mobile_manipulator, load_robot_spec
from physlab.robots.models import RobotKind, SimSource

_ROOT = Path(__file__).resolve().parents[2]


def _catalog() -> RobotCatalog:
    return RobotCatalog(_ROOT / "configs", _ROOT / "assets")


def test_catalog_loads_every_yaml() -> None:
    catalog = _catalog()
    specs = catalog.list()
    stems = sorted(path.stem for path in (_ROOT / "configs" / "robots").glob("*.yaml"))
    assert [spec.id for spec in specs] == stems
    arms = [spec for spec in specs if spec.kind is RobotKind.ARM and spec.available]
    mobile = [
        spec
        for spec in specs
        if spec.kind in {RobotKind.MOBILE_BASE, RobotKind.MOBILE_MANIPULATOR}
        and spec.sim.source is SimSource.MENAGERIE
        and spec.available
    ]
    procedural = [spec for spec in specs if spec.sim.source is SimSource.PROCEDURAL]
    assert len(arms) >= 8
    assert len(mobile) >= 2
    assert {spec.id for spec in procedural} == {"generic_diffdrive", "gantry_xyz"}
    assert all(spec.available for spec in procedural)
    with pytest.raises(ConfigError, match="missing"):
        catalog.get("missing")


def test_procedural_robots_compile() -> None:
    catalog = _catalog()
    for robot_id in ("generic_diffdrive", "gantry_xyz"):
        result = catalog.check(robot_id)
        assert result.ok, result.messages
        model = load_robot_spec(catalog.get(robot_id), _ROOT / "assets")
        compiled = model.compile()
        assert compiled.nu > 0
        assert compiled.njnt > 0


@pytest.mark.assets
def test_every_available_robot_checks() -> None:
    catalog = _catalog()
    for spec in catalog.list():
        if not spec.available:
            continue
        result = catalog.check(spec.id)
        assert result.ok, (spec.id, result.messages)


@pytest.mark.assets
def test_panda_reports_runtime_site() -> None:
    result = _catalog().check("franka_emika_panda")
    assert result.ok
    assert "site will be added at runtime" in result.messages


@pytest.mark.assets
def test_mobile_manipulator_composition_joint_count() -> None:
    catalog = _catalog()
    assets = _ROOT / "assets"
    base = catalog.get("generic_diffdrive")
    base_joints = sum(1 for joint in load_robot_spec(base, assets).joints if joint.name)
    for arm_id in ("trs_so_arm100", "ufactory_xarm7"):
        arm = catalog.get(arm_id)
        arm_joints = sum(1 for joint in load_robot_spec(arm, assets).joints if joint.name)
        composed, model = compose_mobile_manipulator(base, arm, f"diff_{arm_id}", assets)
        compiled = model.compile()
        assert compiled.njnt == base_joints + arm_joints
        assert composed.kind is RobotKind.MOBILE_MANIPULATOR
        assert composed.sim.arm_joints == arm.sim.arm_joints
        assert composed.sim.base_joints == base.sim.base_joints
