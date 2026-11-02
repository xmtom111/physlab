"""Skills an experiment procedure is allowed to name.

The registry holds the motion skills from the arm stack, the controller
runner, and the experiment plants. A procedure step whose skill is missing
is a validation error.
"""

from __future__ import annotations

from collections.abc import Callable

from physlab.assembly.skills import pick, place
from physlab.controllers.runner import run_controller
from physlab.core.errors import ConfigError
from physlab.experiments.library.incline import InclineBench
from physlab.experiments.library.lens import LensBench
from physlab.experiments.library.pendulum import PendulumBench
from physlab.motion.skills import approach, navigate_and_reach, retreat, safe_transit

Skill = Callable[..., object]


class SkillRegistry:
    """Name to callable map for procedure steps."""

    def __init__(self) -> None:
        """Start empty."""
        self._skills: dict[str, Skill] = {}

    def register(self, name: str, skill: Skill) -> None:
        """Register ``skill`` under ``name``.

        Args:
            name: Procedure name.
            skill: Callable the runner will invoke with a trial context.

        Raises:
            ConfigError: ``name`` is empty or already registered.
        """
        if not name:
            raise ConfigError("Skill name must be non-empty")
        if name in self._skills:
            raise ConfigError("Duplicate skill", details={"name": name})
        self._skills[name] = skill

    def get(self, name: str) -> Skill:
        """Return the skill named ``name``.

        Args:
            name: Procedure name.

        Returns:
            Registered callable.

        Raises:
            ConfigError: ``name`` is not registered.
        """
        skill = self._skills.get(name)
        if skill is None:
            raise ConfigError("Unknown skill", details={"name": name})
        return skill

    def names(self) -> tuple[str, ...]:
        """Return registered names in insertion order.

        Returns:
            Skill names.
        """
        return tuple(self._skills)


def experiment_registry() -> SkillRegistry:
    """Build the registry used by the built-in experiments.

    Returns:
        Registry containing arm skills, the controller runner, and the
        pendulum, incline, and lens plants.
    """
    registry = SkillRegistry()
    for name, skill in _imported_skills():
        registry.register(name, skill)
    _register_plants(registry)
    return registry


def _imported_skills() -> tuple[tuple[str, Skill], ...]:
    return (
        ("safe_transit", safe_transit),
        ("approach", approach),
        ("retreat", retreat),
        ("navigate_and_reach", navigate_and_reach),
        ("pick", pick),
        ("place", place),
        ("run_controller", run_controller),
    )


def _register_plants(registry: SkillRegistry) -> None:
    pendulum = PendulumBench()
    registry.register("set_pendulum_length", pendulum.set_pendulum_length)
    registry.register("pull_and_release", pendulum.pull_and_release)
    registry.register("measure_period", pendulum.measure_period)
    incline = InclineBench()
    registry.register("set_incline_angle", incline.set_incline_angle)
    registry.register("release_block", incline.release_block)
    registry.register("measure_acceleration", incline.measure_acceleration)
    lens = LensBench()
    registry.register("set_object_distance", lens.set_object_distance)
    registry.register("focus_image_distance", lens.focus_image_distance)
