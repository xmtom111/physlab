"""Robot-deployable fine alignment.

``dock`` drives the tool to a knob site. Within 1 mm the tool may command the
mount's slide (5 µm) and hinge (0.1 mrad). ``align`` maximizes a scalar
objective with coordinate search or Nelder-Mead.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import minimize

from physlab.core.errors import ConfigError
from physlab.core.types import FailureReason, MotionResult, Pose
from physlab.hal.interfaces import Robot
from physlab.sim.world import SimWorld

_SLIDE_M = 5e-6
_HINGE_RAD = 1e-4


@dataclass(frozen=True)
class AdjustableDof:
    """One screw the tool can turn.

    Attributes:
        name: Name used by ``adjust`` and ``align``.
        joint: MuJoCo joint name.
        actuator: Position actuator name.
        kind: ``slide`` is meters, ``hinge`` is radians.
    """

    name: str
    joint: str
    actuator: str
    kind: Literal["slide", "hinge"]


class FineAlignTool:
    """Dock onto a knob and drive its adjustment screws."""

    def __init__(self, world: SimWorld, knob_site: str, dofs: list[AdjustableDof]) -> None:
        """Store the mount the tool can drive.

        Args:
            world: Simulation that owns the joints.
            knob_site: Site the tool tip must reach.
            dofs: Screws on that mount.
        """
        self.world = world
        self.knob_site = knob_site
        self.dofs = {item.name: item for item in dofs}
        self.docked = False

    def dock(self, robot: Robot, component: str) -> MotionResult:
        """Move the arm to the knob.

        Args:
            robot: Arm that carries the tool.
            component: Mount being aligned. Recorded for the caller; the knob
                site is the one given at construction.

        Returns:
            Success when the tool tip is within 1 mm of the knob.
        """
        del component
        if robot.arm is None:
            raise ConfigError("Fine alignment needs an arm")
        site = self.world.site_pose(self.knob_site)
        current = robot.arm.get_ee_pose()
        goal = Pose(position_m=site.position_m, quat_wxyz=current.quat_wxyz, frame="world")
        moved = robot.arm.move_pose(goal, speed_scale=1.0, timeout_s=12.0)
        distance = _distance(robot.arm.get_ee_pose().position_m, site.position_m)
        for _try in range(3):
            if not moved.success or distance <= 0.001:
                break
            moved = robot.arm.move_pose(goal, speed_scale=1.0, timeout_s=8.0)
            distance = _distance(robot.arm.get_ee_pose().position_m, site.position_m)
        self.docked = bool(moved.success and distance <= 0.001)
        if self.docked:
            return moved
        return MotionResult(
            success=False,
            reason=FailureReason.TOLERANCE,
            message=f"tool is {distance:.4f} m from the knob",
            final_pos_error_m=distance,
            final_rot_error_rad=moved.final_rot_error_rad,
            duration_s=moved.duration_s,
        )

    def adjust(self, dof: str, delta: float) -> None:
        """Turn one screw by ``delta``.

        Args:
            dof: Screw name.
            delta: Meters or radians, quantized to the screw resolution.

        Raises:
            ConfigError: The tool is not docked, or ``dof`` is unknown.
        """
        if not self.docked:
            raise ConfigError("Fine-align tool is not docked")
        screw = self._screw(dof)
        self._write(screw, self.world.joint_position(screw.joint) + delta)

    def undock(self) -> None:
        """Release the screws. Further ``adjust`` calls are rejected."""
        self.docked = False

    def align(
        self,
        objective_fn: Callable[[], float],
        dofs: list[str],
        method: Literal["coordinate_search", "nelder_mead"] = "coordinate_search",
        max_iters: int = 40,
    ) -> float:
        """Maximize ``objective_fn`` by turning ``dofs``.

        Args:
            objective_fn: Score to maximize. Typically an image metric.
            dofs: Screw names.
            method: ``coordinate_search`` or ``nelder_mead``.
            max_iters: Evaluation budget.

        Returns:
            Best score found.

        Raises:
            ConfigError: The tool is not docked or ``method`` is unknown.
        """
        if not self.docked:
            raise ConfigError("Fine-align tool is not docked")
        if method == "coordinate_search":
            return _coordinate_search(self, objective_fn, dofs, max_iters)
        if method == "nelder_mead":
            return _nelder_mead(self, objective_fn, dofs, max_iters)
        raise ConfigError("Unknown alignment method", details={"method": method})

    def _screw(self, name: str) -> AdjustableDof:
        screw = self.dofs.get(name)
        if screw is None:
            raise ConfigError("Unknown adjustment dof", details={"dof": name})
        return screw

    def _write(self, screw: AdjustableDof, value: float) -> None:
        step = _SLIDE_M if screw.kind == "slide" else _HINGE_RAD
        quantized = round(value / step) * step
        low, high = _limits(self.world, screw.joint)
        clipped = min(max(quantized, low), high)
        self.world.set_actuator_ctrl(screw.actuator, clipped)
        self.world.set_joint_position(screw.joint, clipped)
        self.world.project_welds()


def _coordinate_search(
    tool: FineAlignTool,
    objective_fn: Callable[[], float],
    dofs: list[str],
    max_iters: int,
) -> float:
    best = objective_fn()
    evals = 1
    for step in (5e-4, 2.5e-4, 1.25e-4, 6e-5, 3e-5, 1.5e-5, 5e-6):
        for name in dofs:
            screw = tool._screw(name)
            for _walk in range(6):
                if evals >= max_iters:
                    return best
                current = tool.world.joint_position(screw.joint)
                improved, best, used = _try_step(tool, objective_fn, screw, current, step, best)
                evals += used
                if not improved:
                    break
    return best


def _try_step(
    tool: FineAlignTool,
    objective_fn: Callable[[], float],
    screw: AdjustableDof,
    current: float,
    step: float,
    best: float,
) -> tuple[bool, float, int]:
    """Try one step either way. Keep it when the score rises."""
    for sign in (1.0, -1.0):
        tool._write(screw, current + sign * step)
        score = objective_fn()
        if score > best:
            return True, score, 1 if sign > 0 else 2
        tool._write(screw, current)
    return False, best, 2


def _nelder_mead(
    tool: FineAlignTool,
    objective_fn: Callable[[], float],
    dofs: list[str],
    max_iters: int,
) -> float:
    start = np.array([tool.world.joint_position(tool._screw(name).joint) for name in dofs])
    simplex = [start]
    for index in range(len(dofs)):
        offset = start.copy()
        offset[index] -= 0.001
        simplex.append(offset)

    def _cost(values: np.ndarray) -> float:
        for name, value in zip(dofs, values, strict=True):
            tool._write(tool._screw(name), float(value))
        return -objective_fn()

    solved = minimize(
        _cost,
        start,
        method="Nelder-Mead",
        options={"maxiter": max_iters, "xatol": _SLIDE_M, "initial_simplex": np.vstack(simplex)},
    )
    _cost(np.asarray(solved.x, dtype=np.float64))
    return -float(solved.fun)


def _limits(world: SimWorld, joint: str) -> tuple[float, float]:
    import mujoco

    joint_id = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    low, high = world.model.jnt_range[joint_id]
    return float(low), float(high)


def _distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return float(np.linalg.norm(np.asarray(left) - np.asarray(right)))
