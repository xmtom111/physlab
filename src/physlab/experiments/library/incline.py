"""Block on an incline.

The block is constrained to a slide along the ramp. Joint friction loss is
``mu * mass * g * cos(theta)``, which is the Coulomb force from the sim
friction coefficient. Acceleration is the quadratic coefficient of the
tracked position.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from physlab.experiments.protocol import (
    ExperimentProtocol,
    Measurement,
    ProcedureStep,
    SafetyClass,
    Variable,
)

SIM_GRAVITY = 9.81
SIM_FRICTION = 0.1
_MASS_KG = 0.2
_DT_S = 0.001


class InclineBench:
    """One incline reused across angles."""

    def __init__(self) -> None:
        """Start at 20 degrees."""
        self.angle_deg = 20.0

    def set_incline_angle(self, ctx: object) -> None:
        """Read ``angle_deg`` from the trial.

        Args:
            ctx: Trial context.
        """
        self.angle_deg = _value(ctx, "angle_deg")

    def release_block(self, ctx: object) -> None:
        """Mark the block released. Measurement performs the slide.

        Args:
            ctx: Unused. The slide starts from rest.
        """
        del ctx

    def measure_acceleration(self, ctx: object) -> None:
        """Record ``accel_m_s2`` from the tracked slide.

        Args:
            ctx: Trial context that receives the measurement.
        """
        _records(ctx)["accel_m_s2"] = incline_acceleration(self.angle_deg)


def incline_protocol(seed: int = 0) -> ExperimentProtocol:
    """Return the angle sweep.

    Args:
        seed: Shuffle seed.

    Returns:
        Protocol E2.
    """
    return ExperimentProtocol(
        id="E2_incline",
        question="How does acceleration down a ramp depend on angle?",
        hypothesis="a = g (sin theta - mu cos theta).",
        apparatus="incline_ramp",
        independent=[Variable(name="angle_deg", unit="deg", values=[10, 15, 20, 25, 30, 35])],
        dependent=[
            Measurement(
                name="accel_m_s2",
                unit="m/s^2",
                method="quadratic fit of tracked position along the ramp",
            )
        ],
        controlled={"mu": SIM_FRICTION},
        procedure=[
            ProcedureStep(skill="set_incline_angle"),
            ProcedureStep(skill="release_block"),
            ProcedureStep(skill="measure_acceleration"),
        ],
        repetitions=1,
        seed=seed,
        safety_class=SafetyClass.SIM_ONLY,
    )


def incline_acceleration(angle_deg: float) -> float:
    """Return the measured downhill acceleration.

    Args:
        angle_deg: Ramp angle, degrees.

    Returns:
        Acceleration, meters per second squared.
    """
    model, data = _compile(angle_deg)
    count = int(1.0 / _DT_S)
    times = np.empty(count, dtype=np.float64)
    position = np.empty(count, dtype=np.float64)
    for index in range(count):
        mujoco.mj_step(model, data)
        times[index] = data.time
        position[index] = float(data.qpos[0])
    mask = times > 0.2
    clock = times[mask] - times[mask][0]
    coef = np.polyfit(clock, position[mask], 2)
    return float(2.0 * coef[0])


def _compile(angle_deg: float) -> tuple[mujoco.MjModel, mujoco.MjData]:
    angle = math.radians(angle_deg)
    spec = mujoco.MjSpec()
    spec.option.timestep = _DT_S
    spec.option.gravity[:] = [0.0, 0.0, -SIM_GRAVITY]
    half = angle / 2.0
    rail = spec.worldbody.add_body(name="rail", quat=[math.cos(half), 0.0, math.sin(half), 0.0])
    block = rail.add_body(name="block")
    friction = SIM_FRICTION * _MASS_KG * SIM_GRAVITY * math.cos(angle)
    block.add_joint(
        name="slide",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[1.0, 0.0, 0.0],
        frictionloss=friction,
    )
    block.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.01], mass=_MASS_KG)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _value(ctx: object, name: str) -> float:
    return float(getattr(ctx, "values", {})[name])


def _records(ctx: object) -> dict[str, float]:
    records = getattr(ctx, "records", None)
    if not isinstance(records, dict):
        raise RuntimeError("trial context has no records")
    return records
