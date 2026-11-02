"""Stepper-driven focus stage.

``stage_rate`` is steps per second. One revolution moves the stage 1 mm.
``defocus_m`` is the signed distance from the focal plane. ``focus`` is a
sharpness score that peaks at 1 when ``defocus_m`` is 0.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from physlab.controllers.api import ActuatorChannel, DeviceManifest, SensorChannel
from physlab.sim.virtual_devices import StepperMotor

_STEPS_PER_REV = 200.0
_METERS_PER_REV = 0.001
_MAX_STEPS_S = 5000.0


def focus_manifest() -> DeviceManifest:
    """Return the focus-stage channels.

    Returns:
        Defocus, sharpness, and the stepper rate command.
    """
    return DeviceManifest(
        sensors={
            "defocus_m": SensorChannel(unit="m", noise_std=0.0, range=(-0.01, 0.01)),
            "focus": SensorChannel(unit="1", noise_std=0.0, range=(0.0, 1.0)),
        },
        actuators={
            "stage_rate": ActuatorChannel(
                unit="1/s", min=-_MAX_STEPS_S, max=_MAX_STEPS_S, rate_limit=0.0
            )
        },
    )


class FocusStage:
    """Kinematic stage. The focal plane is z = 0."""

    def __init__(self) -> None:
        """Start 1 mm away from focus. ``reset`` rebuilds the stepper."""
        self.manifest = focus_manifest()
        self.stepper = StepperMotor(_STEPS_PER_REV, _MAX_STEPS_S, seed=0)
        self.z_m = 0.001

    def reset(self, seed: int) -> None:
        """Place the stage 1 mm positive of focus.

        Args:
            seed: Stepper noise seed. The scenario sensors are noiseless.
        """
        self.stepper = StepperMotor(_STEPS_PER_REV, _MAX_STEPS_S, seed=seed)
        self.z_m = 0.001
        self.stepper.position_steps = self.z_m / _METERS_PER_REV * _STEPS_PER_REV

    def measure(self) -> dict[str, float]:
        """Return defocus and a sharpness score.

        Returns:
            ``defocus_m`` in meters and ``focus`` in ``[0, 1]``.
        """
        defocus = self.z_m
        sharpness = math.exp(-((defocus / 50e-6) ** 2))
        return {"defocus_m": defocus, "focus": min(sharpness, 1.0)}

    def apply(self, action: Mapping[str, float], dt_s: float) -> None:
        """Integrate the stepper and update stage height.

        Args:
            action: Clipped commands. ``stage_rate`` is steps per second.
            dt_s: Tick length, seconds.
        """
        self.stepper.command_rate(float(action.get("stage_rate", 0.0)), dt_s)
        self.z_m = self.stepper.revolutions * _METERS_PER_REV
