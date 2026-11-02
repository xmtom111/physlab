"""Servo that holds a pendulum bob, then lets it swing.

``hold`` at least 0.5 locks the bob on the servo angle. Below 0.5 the hinge
is free and the bob falls under gravity. ``angle_rad`` is the bob angle from
vertical.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from physlab.controllers.api import ActuatorChannel, DeviceManifest, SensorChannel
from physlab.sim.virtual_devices import Servo

_LENGTH_M = 0.25
_GRAVITY = 9.81


def pendulum_manifest() -> DeviceManifest:
    """Return the pendulum channels.

    Returns:
        Bob angle, servo angle command, and the hold flag.
    """
    return DeviceManifest(
        sensors={"angle_rad": SensorChannel(unit="rad", range=(-math.pi, math.pi))},
        actuators={
            "servo_rad": ActuatorChannel(unit="rad", min=-1.5, max=1.5, rate_limit=2.0),
            "hold": ActuatorChannel(unit="1", min=0.0, max=1.0, rate_limit=0.0),
        },
    )


class PendulumRelease:
    """Planar pendulum. The servo is the latch."""

    def __init__(self) -> None:
        """Hold the bob at 0.4 rad."""
        self.manifest = pendulum_manifest()
        self.servo = Servo(2.0, seed=0)
        self.angle_rad = 0.4
        self.omega_rad_s = 0.0
        self.holding = True

    def reset(self, seed: int) -> None:
        """Re-latch the bob at 0.4 rad.

        Args:
            seed: Servo noise seed.
        """
        self.servo = Servo(2.0, seed=seed)
        self.servo.angle_rad = 0.4
        self.angle_rad = 0.4
        self.omega_rad_s = 0.0
        self.holding = True

    def measure(self) -> dict[str, float]:
        """Return the bob angle.

        Returns:
            ``angle_rad``.
        """
        return {"angle_rad": self.angle_rad}

    def apply(self, action: Mapping[str, float], dt_s: float) -> None:
        """Slew the servo and either lock or integrate the pendulum.

        Args:
            action: ``servo_rad`` and ``hold``.
            dt_s: Tick length, seconds.
        """
        self.servo.command_angle(float(action.get("servo_rad", self.servo.angle_rad)), dt_s)
        self.holding = float(action.get("hold", 1.0)) >= 0.5
        if self.holding:
            self.angle_rad = self.servo.angle_rad
            self.omega_rad_s = 0.0
            return
        accel = -(_GRAVITY / _LENGTH_M) * math.sin(self.angle_rad)
        self.omega_rad_s += accel * dt_s
        self.angle_rad += self.omega_rad_s * dt_s
