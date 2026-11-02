"""Differential-drive base and two downward photodiodes over a bright line.

The line is the strip ``|y| < 1 cm`` in the world. Wheel commands are
radians per second. The wheel radius is 3 cm and the track is 10 cm.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from physlab.controllers.api import ActuatorChannel, DeviceManifest, SensorChannel
from physlab.sim.virtual_devices import Photodiode

_RADIUS_M = 0.03
_TRACK_M = 0.10
_SENSOR_Y_M = 0.015
_LINE_HALF_M = 0.01


def line_manifest() -> DeviceManifest:
    """Return the line-follower channels.

    Returns:
        Two photodiodes and two wheel-speed commands.
    """
    span = ActuatorChannel(unit="rad/s", min=-20.0, max=20.0, rate_limit=0.0)
    eye = SensorChannel(unit="1", range=(0.0, 1.0))
    return DeviceManifest(
        sensors={"photo_left": eye, "photo_right": eye},
        actuators={"left_rad_s": span, "right_rad_s": span},
    )


class LineFollow:
    """Unicycle kinematics with sensors beside the chassis center."""

    def __init__(self) -> None:
        """Start just beside the line, heading along +X."""
        self.manifest = line_manifest()
        self.x_m = 0.0
        self.y_m = 0.02
        self.yaw_rad = 0.0
        self.left = Photodiode(seed=0)
        self.right = Photodiode(seed=1)

    def reset(self, seed: int) -> None:
        """Restore the start pose and reseed the photodiodes.

        Args:
            seed: Base seed. The right eye uses ``seed + 1``.
        """
        self.x_m = 0.0
        self.y_m = 0.02
        self.yaw_rad = 0.0
        self.left = Photodiode(seed=seed)
        self.right = Photodiode(seed=seed + 1)

    def measure(self) -> dict[str, float]:
        """Return both photodiode intensities.

        Returns:
            ``photo_left`` and ``photo_right`` in ``[0, 1]``.
        """
        return {
            "photo_left": self.left.read(self._intensity(-_SENSOR_Y_M)),
            "photo_right": self.right.read(self._intensity(_SENSOR_Y_M)),
        }

    def apply(self, action: Mapping[str, float], dt_s: float) -> None:
        """Integrate wheel speeds into the chassis pose.

        Args:
            action: ``left_rad_s`` and ``right_rad_s``.
            dt_s: Tick length, seconds.
        """
        left = float(action.get("left_rad_s", 0.0)) * _RADIUS_M
        right = float(action.get("right_rad_s", 0.0)) * _RADIUS_M
        forward = 0.5 * (left + right)
        yaw_rate = (right - left) / _TRACK_M
        self.yaw_rad += yaw_rate * dt_s
        self.x_m += forward * math.cos(self.yaw_rad) * dt_s
        self.y_m += forward * math.sin(self.yaw_rad) * dt_s

    def _intensity(self, body_y_m: float) -> float:
        world_y = self.y_m + body_y_m * math.cos(self.yaw_rad)
        offset = abs(world_y) - _LINE_HALF_M
        if offset <= 0.0:
            return 1.0
        return math.exp(-offset / 0.005)
