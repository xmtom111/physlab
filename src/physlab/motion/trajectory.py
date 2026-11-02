"""Synchronized trapezoidal time scaling for joint waypoints."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from physlab.core.errors import ConfigError

_MIN_SPAN = 1e-9


@dataclass(frozen=True)
class _Segment:
    t_acc: float
    t_cruise: float
    accel: float
    velocity: float
    distance_acc: float


class Trajectory:
    """Joint waypoints with a shared trapezoidal clock.

    Attributes:
        times_s: Time at each waypoint, starting at 0. The last entry is the
            duration.
        q: Waypoint positions, shape ``(n_waypoints, dof)``.
    """

    def __init__(self, times_s: np.ndarray, q: np.ndarray, segments: tuple[_Segment, ...]) -> None:
        """Store the sampled path.

        Args:
            times_s: Cumulative time at each waypoint.
            q: Waypoint array.
            segments: One trapezoid per interval between waypoints.
        """
        self.times_s = times_s
        self.q = q
        self._segments = segments

    @property
    def duration_s(self) -> float:
        """Total duration, seconds."""
        return float(self.times_s[-1])

    def sample(self, time_s: float) -> np.ndarray:
        """Return the joint command at ``time_s``.

        Args:
            time_s: Time from the start of the trajectory, seconds.

        Returns:
            Joint positions. Times before 0 and after the duration clamp.
        """
        if time_s <= 0.0 or self.q.shape[0] == 1:
            return np.array(self.q[0], dtype=np.float64, copy=True)
        if time_s >= self.duration_s:
            return np.array(self.q[-1], dtype=np.float64, copy=True)
        index = int(np.searchsorted(self.times_s, time_s, side="right") - 1)
        index = min(max(index, 0), len(self._segments) - 1)
        local = time_s - float(self.times_s[index])
        fraction = _path_fraction(self._segments[index], local)
        start = self.q[index]
        end = self.q[index + 1]
        return np.asarray(start + fraction * (end - start), dtype=np.float64)


def time_parameterize(
    waypoints: Sequence[Sequence[float]],
    vel_limits: Sequence[float],
    acc_limits: Sequence[float],
    speed_scale: float,
) -> Trajectory:
    """Build a synchronized trapezoidal trajectory through ``waypoints``.

    Each joint uses the same path parameter. Velocity and acceleration limits
    are scaled by ``speed_scale``, and the segment time is set by the slowest
    joint.

    Args:
        waypoints: Joint positions. At least one waypoint.
        vel_limits: Per-joint speed limits, radians per second.
        acc_limits: Per-joint acceleration limits, radians per second squared.
        speed_scale: Positive fraction applied to both limits.

    Returns:
        Trajectory whose ``sample(0)`` is the first waypoint.

    Raises:
        ConfigError: The inputs are empty, mismatched, or not positive.
    """
    if speed_scale <= 0.0:
        raise ConfigError("speed_scale must be positive", details={"speed_scale": speed_scale})
    rows = [list(map(float, point)) for point in waypoints]
    if not rows:
        raise ConfigError("Trajectory needs at least one waypoint")
    dof = len(rows[0])
    velocity = _limits(vel_limits, dof, "vel_limits") * speed_scale
    accel = _limits(acc_limits, dof, "acc_limits") * speed_scale
    q = np.asarray(rows, dtype=np.float64)
    times = [0.0]
    segments: list[_Segment] = []
    for start, end in itertools.pairwise(q):
        segment, duration = _trapezoid(start, end, velocity, accel)
        segments.append(segment)
        times.append(times[-1] + duration)
    return Trajectory(np.asarray(times, dtype=np.float64), q, tuple(segments))


def _limits(values: Sequence[float], dof: int, name: str) -> np.ndarray:
    limits = np.asarray([float(item) for item in values], dtype=np.float64)
    if limits.shape != (dof,) or np.any(limits <= 0.0) or not np.all(np.isfinite(limits)):
        raise ConfigError(
            f"{name} must be positive and match the waypoint dof",
            details={"dof": dof, "length": int(limits.size)},
        )
    return limits


def _trapezoid(
    start: np.ndarray,
    end: np.ndarray,
    velocity: np.ndarray,
    accel: np.ndarray,
) -> tuple[_Segment, float]:
    delta = np.abs(end - start)
    moving = delta > _MIN_SPAN
    if not np.any(moving):
        return _Segment(0.0, 0.0, 0.0, 0.0, 0.0), 0.0
    v_path = float(np.min(velocity[moving] / delta[moving]))
    a_path = float(np.min(accel[moving] / delta[moving]))
    if v_path * v_path / a_path >= 1.0:
        t_acc = float(np.sqrt(1.0 / a_path))
        return _Segment(t_acc, 0.0, a_path, a_path * t_acc, 0.5), 2.0 * t_acc
    t_acc = v_path / a_path
    distance_acc = 0.5 * a_path * t_acc * t_acc
    t_cruise = (1.0 - 2.0 * distance_acc) / v_path
    duration = 2.0 * t_acc + t_cruise
    return _Segment(t_acc, t_cruise, a_path, v_path, distance_acc), duration


def _path_fraction(segment: _Segment, time_s: float) -> float:
    if segment.t_acc <= 0.0 and segment.t_cruise <= 0.0:
        return 1.0
    if time_s <= segment.t_acc:
        return 0.5 * segment.accel * time_s * time_s
    cruise_end = segment.t_acc + segment.t_cruise
    if time_s <= cruise_end:
        return segment.distance_acc + segment.velocity * (time_s - segment.t_acc)
    tail = time_s - cruise_end
    coast = segment.distance_acc + segment.velocity * segment.t_cruise
    return coast + segment.velocity * tail - 0.5 * segment.accel * tail * tail
