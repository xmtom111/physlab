"""Load ``configs/robots/*.yaml`` and check the names against MuJoCo."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import mujoco
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from physlab.core.errors import AssetMissing, ConfigError
from physlab.robots.compose import element_exists, gripper_prefix, load_robot_spec
from physlab.robots.models import RobotSpec

_SITE_NOTE = "site will be added at runtime"


class CheckResult(BaseModel):
    """Outcome of :meth:`RobotCatalog.check`.

    Attributes:
        robot_id: Catalog id that was checked.
        ok: True when every listed name exists and the spec compiles.
        messages: Errors, or the runtime-site note. Empty when nothing to say.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    robot_id: str
    ok: bool
    messages: tuple[str, ...] = ()


class RobotCatalog:
    """Robots declared as YAML files in one directory."""

    def __init__(self, configs_dir: Path, assets_dir: Path) -> None:
        """Load every ``configs_dir/robots/*.yaml`` file.

        Args:
            configs_dir: Directory whose ``robots`` subdirectory holds the YAML.
            assets_dir: Directory passed to the MuJoCo loader.

        Raises:
            ConfigError: A YAML file is not a valid :class:`RobotSpec`, or two
                files share an id.
        """
        self._assets_dir = assets_dir
        self._robots = _load_dir(configs_dir / "robots")

    def list(self) -> list[RobotSpec]:
        """Return every robot, sorted by id.

        Returns:
            Catalog entries. The list is a copy.
        """
        return [self._robots[key] for key in sorted(self._robots)]

    def get(self, robot_id: str) -> RobotSpec:
        """Return the robot named ``robot_id``.

        Args:
            robot_id: Catalog id.

        Returns:
            The matching spec.

        Raises:
            ConfigError: No robot uses that id.
        """
        found = self._robots.get(robot_id)
        if found is None:
            raise ConfigError(
                f"Unknown robot {robot_id!r}",
                details={"robot": robot_id, "known": sorted(self._robots)},
            )
        return found

    def check(self, robot_id: str) -> CheckResult:
        """Load ``robot_id`` and confirm the names in its spec exist.

        A missing ``ee_site`` with an ``ee_body`` is not a failure. The result
        message is ``site will be added at runtime``. ``home_qpos_rad`` must
        have one value per arm joint. Unavailable robots are reported and are
        not loaded.

        Args:
            robot_id: Catalog id.

        Returns:
            Pass or fail, with messages.

        Raises:
            ConfigError: ``robot_id`` is not in the catalog.
        """
        spec = self.get(robot_id)
        if not spec.available:
            reason = spec.unavailable_reason or "unavailable"
            return CheckResult(robot_id=robot_id, ok=False, messages=(reason,))
        try:
            model = load_robot_spec(spec, self._assets_dir)
            model.compile()
        except (AssetMissing, ConfigError) as exc:
            return CheckResult(robot_id=robot_id, ok=False, messages=(str(exc),))
        except (ValueError, OSError, RuntimeError) as exc:
            return CheckResult(
                robot_id=robot_id,
                ok=False,
                messages=(f"compile failed: {exc}",),
            )
        problems = _problems(spec, model)
        notes = _notes(spec)
        return CheckResult(
            robot_id=robot_id,
            ok=not problems,
            messages=tuple([*notes, *problems]),
        )


def _load_dir(directory: Path) -> dict[str, RobotSpec]:
    if not directory.is_dir():
        raise ConfigError("Robot config directory was not found", details={"path": str(directory)})
    robots: dict[str, RobotSpec] = {}
    for path in sorted(directory.glob("*.yaml")):
        spec = _load_file(path)
        if spec.id in robots:
            raise ConfigError("Duplicate robot id", details={"robot": spec.id, "path": str(path)})
        if path.stem != spec.id:
            raise ConfigError(
                "Robot YAML file name must match its id",
                details={"path": str(path), "id": spec.id},
            )
        robots[spec.id] = spec
    return robots


def _load_file(path: Path) -> RobotSpec:
    loaded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        return RobotSpec.model_validate(loaded)
    except ValidationError as exc:
        raise ConfigError(
            "Robot YAML is invalid",
            details={"path": str(path), "reason": str(exc)},
        ) from exc


def _notes(spec: RobotSpec) -> list[str]:
    if spec.sim.ee_site is None and spec.sim.ee_body is not None:
        return [_SITE_NOTE]
    return []


def _problems(spec: RobotSpec, model: mujoco.MjSpec) -> list[str]:
    problems: list[str] = []
    sim = spec.sim
    prefix = gripper_prefix(spec)
    _require(model, "joint", sim.arm_joints, problems)
    _require(model, "joint", sim.base_joints, problems)
    _require(model, "actuator", sim.arm_actuators, problems)
    _require(model, "actuator", sim.base_actuators, problems)
    if sim.mount_site is not None:
        _require(model, "site", (sim.mount_site,), problems)
    if sim.ee_body is not None:
        _require(model, "body", (sim.ee_body,), problems)
    if sim.ee_site is not None and not _site_present(model, sim.ee_site, prefix):
        problems.append(f"missing site {sim.ee_site}")
    if sim.gripper is not None:
        actuator = f"{prefix}{sim.gripper.actuator}"
        fingers = tuple(f"{prefix}{name}" for name in sim.gripper.finger_geoms)
        _require(model, "actuator", (actuator,), problems)
        _require(model, "geom", fingers, problems)
    if len(sim.home_qpos_rad) != len(sim.arm_joints):
        problems.append(
            "home_qpos_rad length "
            f"{len(sim.home_qpos_rad)} does not match arm_joints length {len(sim.arm_joints)}"
        )
    return problems


def _site_present(model: mujoco.MjSpec, name: str, prefix: str) -> bool:
    if element_exists(model, "site", name):
        return True
    return bool(prefix) and element_exists(model, "site", f"{prefix}{name}")


def _require(
    model: mujoco.MjSpec,
    kind: str,
    names: tuple[str, ...],
    problems: list[str],
) -> None:
    for name in names:
        if not element_exists(model, kind, name):
            problems.append(f"missing {kind} {name}")
