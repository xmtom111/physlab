"""Pendulum period versus length.

A slide joint sets the bob distance. A position servo holds that length.
The hinge is pulled to 10 degrees and released. The period is the mean gap
between positive-going zero crossings of the bob's horizontal position, which
is the track an overhead ArUco marker would give.
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
_DT_S = 0.001
_DURATION_S = 6.0


class PendulumBench:
    """One MuJoCo pendulum reused across trials."""

    def __init__(self) -> None:
        """Start at 0.3 m. ``set_pendulum_length`` overwrites it."""
        self.length_m = 0.3
        self.release_rad = math.radians(10.0)

    def set_pendulum_length(self, ctx: object) -> None:
        """Read ``length_m`` from the trial.

        Args:
            ctx: Trial context. Uses ``values['length_m']``.
        """
        self.length_m = _value(ctx, "length_m")

    def pull_and_release(self, ctx: object) -> None:
        """Store the release angle. The next measurement lets the bob go.

        Args:
            ctx: Trial context. ``release_deg`` defaults to 10.
        """
        degrees = _value(ctx, "release_deg") if _has(ctx, "release_deg") else 10.0
        self.release_rad = math.radians(degrees)

    def measure_period(self, ctx: object) -> None:
        """Simulate the swing and record ``period_s``.

        Args:
            ctx: Trial context that receives the measurement.
        """
        _records(ctx)["period_s"] = pendulum_period(self.length_m, self.release_rad)


def pendulum_protocol(seed: int = 0) -> ExperimentProtocol:
    """Return the length-versus-period protocol.

    Args:
        seed: Shuffle seed.

    Returns:
        Protocol E1.
    """
    lengths = [0.2, 0.3, 0.45, 0.6, 0.8]
    return ExperimentProtocol(
        id="E1_pendulum",
        question="How does pendulum period depend on length?",
        hypothesis="Period follows a power law with exponent 1/2.",
        apparatus="pendulum_kit",
        independent=[Variable(name="length_m", unit="m", values=lengths)],
        dependent=[
            Measurement(
                name="period_s",
                unit="s",
                method="positive-going zero crossings of the bob marker",
            )
        ],
        controlled={"release_deg": 10.0},
        procedure=[
            ProcedureStep(skill="set_pendulum_length"),
            ProcedureStep(skill="pull_and_release"),
            ProcedureStep(skill="measure_period"),
        ],
        repetitions=1,
        seed=seed,
        safety_class=SafetyClass.SIM_ONLY,
    )


def pendulum_period(length_m: float, release_rad: float) -> float:
    """Return the measured period of a released pendulum.

    Args:
        length_m: Bob distance from the pivot, meters.
        release_rad: Starting angle from vertical, radians.

    Returns:
        Mean period, seconds.

    Raises:
        RuntimeError: Fewer than two full swings were observed.
    """
    model, data = _compile(length_m, release_rad)
    times, lateral = _trace(model, data)
    crossings = _positive_crossings(times, lateral)
    gaps = np.diff(crossings)
    if gaps.size < 2:
        raise RuntimeError("pendulum did not complete two swings")
    return float(np.mean(gaps[1:]))


def _compile(length_m: float, release_rad: float) -> tuple[mujoco.MjModel, mujoco.MjData]:
    spec = mujoco.MjSpec()
    spec.option.timestep = _DT_S
    spec.option.gravity[:] = [0.0, 0.0, -SIM_GRAVITY]
    rod = spec.worldbody.add_body(name="rod")
    rod.add_joint(name="hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0.0, 1.0, 0.0], damping=0.0)
    rod.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.004], mass=0.001, pos=[0.0, 0.0, 0.0])
    bob = rod.add_body(name="bob")
    bob.add_joint(
        name="length",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0.0, 0.0, -1.0],
        limited=1,
        range=[0.05, 1.0],
    )
    bob.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01], mass=0.2, pos=[0.0, 0.0, 0.0])
    _length_servo(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    _set(model, data, "hinge", release_rad)
    _set(model, data, "length", length_m)
    data.ctrl[_actuator(model, "length")] = length_m
    mujoco.mj_forward(model, data)
    return model, data


def _length_servo(spec: mujoco.MjSpec) -> None:
    spec.add_actuator(
        name="length",
        type=mujoco.mjtActuator.mjACTUATOR_POSITION,
        target="length",
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_AFFINE,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        gainprm=[500.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        biasprm=[0.0, -500.0, -20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ctrllimited=1,
        ctrlrange=[0.05, 1.0],
    )


def _trace(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    bob = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bob")
    count = int(_DURATION_S / _DT_S)
    times = np.empty(count, dtype=np.float64)
    lateral = np.empty(count, dtype=np.float64)
    for index in range(count):
        mujoco.mj_step(model, data)
        times[index] = data.time
        lateral[index] = float(data.xpos[bob][0])
    return times, lateral


def _positive_crossings(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    found: list[float] = []
    for index in range(1, values.size):
        previous = float(values[index - 1])
        current = float(values[index])
        if previous <= 0.0 < current:
            gap = current - previous
            frac = 0.0 if gap == 0.0 else -previous / gap
            step = float(times[index] - times[index - 1])
            found.append(float(times[index - 1]) + frac * step)
    return np.asarray(found, dtype=np.float64)


def _set(model: mujoco.MjModel, data: mujoco.MjData, joint: str, value: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    data.qpos[int(model.jnt_qposadr[joint_id])] = value


def _actuator(model: mujoco.MjModel, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))


def _value(ctx: object, name: str) -> float:
    values = getattr(ctx, "values", {})
    return float(values[name])


def _has(ctx: object, name: str) -> bool:
    values = getattr(ctx, "values", {})
    return name in values


def _records(ctx: object) -> dict[str, float]:
    records = getattr(ctx, "records", None)
    if not isinstance(records, dict):
        raise RuntimeError("trial context has no records")
    return records
