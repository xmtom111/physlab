"""The only API controller programs may import from physlab.

A controller sees named sensor readings and returns named actuator commands.
It does not receive the simulator, the filesystem, or a clock it can sleep on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class SensorChannel(BaseModel):
    """One sensor the controller can read.

    Attributes:
        unit: SI unit or ``1`` for a dimensionless reading.
        noise_std: Standard deviation added by the runner, in ``unit``.
        range: Inclusive limits applied after noise.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    unit: str
    noise_std: float = Field(default=0.0, ge=0.0)
    range: tuple[float, float] = (-1.0e9, 1.0e9)


class ActuatorChannel(BaseModel):
    """One actuator command the runner will clip.

    Attributes:
        unit: SI unit of the command.
        min: Lowest command the plant will accept.
        max: Highest command the plant will accept.
        rate_limit: Maximum change per second. A non-positive value means
            the runner does not apply a rate limit.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    unit: str
    min: float
    max: float
    rate_limit: float = 0.0


class DeviceManifest(BaseModel):
    """Sensors and actuators a controller is allowed to name.

    Attributes:
        sensors: Observation keys.
        actuators: Action keys.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    sensors: dict[str, SensorChannel] = Field(default_factory=dict)
    actuators: dict[str, ActuatorChannel] = Field(default_factory=dict)


class ControllerContext:
    """Clock, parameters, and the metric log for one controller process.

    ``log`` records a scalar for the current tick. ``assert_that`` records a
    failure and lets the step continue, so a plot still exists to debug from.
    """

    def __init__(self, dt_s: float, params: Mapping[str, float]) -> None:
        """Store the tick length and a copy of ``params``.

        Args:
            dt_s: Control period, seconds.
            params: Numeric gains and set-points supplied with the run.
        """
        self.time_s = 0.0
        self.dt_s = dt_s
        self.params: Mapping[str, float] = dict(params)
        self._logs: list[tuple[str, float]] = []
        self.assertion_failures: list[tuple[float, str]] = []

    def log(self, key: str, value: float) -> None:
        """Record ``value`` under ``key`` for this tick.

        Args:
            key: Metric name. A sensor name is stored as ``log_<key>``.
            value: Scalar metric.
        """
        self._logs.append((key, float(value)))

    def assert_that(self, cond: bool, message: str) -> None:
        """Record ``message`` when ``cond`` is false.

        Args:
            cond: Condition that should hold.
            message: Text stored with the failure.
        """
        if not cond:
            self.assertion_failures.append((self.time_s, message))

    def drain_logs(self) -> dict[str, float]:
        """Return and clear metrics logged since the last drain.

        Returns:
            Last value written for each key during this tick.
        """
        rows = dict(self._logs)
        self._logs.clear()
        return rows


class Controller(ABC):
    """User controller. Subclasses live in checked source, not in this package."""

    def setup(self, ctx: ControllerContext) -> None:
        """Prepare controller state. The default does nothing.

        Args:
            ctx: Context for the whole run. ``time_s`` is still zero.
        """
        return None

    @abstractmethod
    def step(self, ctx: ControllerContext, obs: Mapping[str, float]) -> Mapping[str, float]:
        """Return actuator commands for this tick.

        Args:
            ctx: Clock and parameters. ``time_s`` is the start of the tick.
            obs: Sensor readings keyed by the manifest.

        Returns:
            Actuator commands keyed by the manifest. Missing keys hold their
            previous command.
        """

    def teardown(self, ctx: ControllerContext) -> None:
        """Release controller state. The default does nothing.

        Args:
            ctx: Context after the last tick.
        """
        return None
