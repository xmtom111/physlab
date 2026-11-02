"""Joint names and qpos addresses for one placed robot."""

from __future__ import annotations

import mujoco
import numpy as np

from physlab.core.errors import ConfigError
from physlab.robots.compose import RUNTIME_EE_SITE
from physlab.robots.models import RobotSpec
from physlab.sim.scene import placed_spec
from physlab.sim.world import SimWorld


def arm_joint_names(world: SimWorld, instance: str) -> tuple[str, ...]:
    """Return the prefixed arm joint names for ``instance``.

    Args:
        world: Simulation that contains the robot.
        instance: Placement prefix.

    Returns:
        Joint names in catalog order.

    Raises:
        ConfigError: ``instance`` is not in the scene or has no arm.
    """
    spec = _spec(world, instance)
    if not spec.sim.arm_joints:
        raise ConfigError("Robot has no arm joints", details={"instance": instance})
    return tuple(f"{instance}/{name}" for name in spec.sim.arm_joints)


def end_effector_site(world: SimWorld, instance: str) -> str:
    """Return the prefixed end-effector site name.

    Args:
        world: Simulation that contains the robot.
        instance: Placement prefix.

    Returns:
        Site name in the compiled model.
    """
    spec = _spec(world, instance)
    site = spec.sim.ee_site or RUNTIME_EE_SITE
    return f"{instance}/{site}"


def qpos_addresses(model: mujoco.MjModel, names: tuple[str, ...]) -> tuple[int, ...]:
    """Return qpos addresses for scalar joints.

    Args:
        model: Compiled model.
        names: Joint names.

    Returns:
        Address of each joint in ``qpos``.

    Raises:
        ConfigError: A joint is missing.
    """
    addresses: list[int] = []
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ConfigError("Joint was not found", details={"joint": name})
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return tuple(addresses)


def read_arm(qpos: np.ndarray, addresses: tuple[int, ...]) -> list[float]:
    """Read arm joint positions from a full configuration.

    Args:
        qpos: Full MuJoCo configuration.
        addresses: Arm qpos addresses.

    Returns:
        Arm joint positions.
    """
    return [float(qpos[address]) for address in addresses]


def write_arm(qpos: np.ndarray, addresses: tuple[int, ...], arm_q: list[float]) -> np.ndarray:
    """Copy ``qpos`` and replace the arm joints.

    Args:
        qpos: Full configuration to copy.
        addresses: Arm qpos addresses.
        arm_q: New arm positions.

    Returns:
        A new configuration vector.
    """
    updated = np.array(qpos, dtype=np.float64, copy=True)
    for address, value in zip(addresses, arm_q, strict=True):
        updated[address] = value
    return updated


def frozen_dofs(model: mujoco.MjModel, names: tuple[str, ...]) -> list[int]:
    """Return every dof that is not one of ``names``.

    Args:
        model: Compiled model.
        names: Joints that IK is allowed to move.

    Returns:
        Dof indices to hold still.
    """
    moving: set[int] = set()
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            continue
        address = int(model.jnt_dofadr[joint_id])
        width = _dof_width(int(model.jnt_type[joint_id]))
        moving.update(range(address, address + width))
    return [dof for dof in range(model.nv) if dof not in moving]


def joint_bounds(model: mujoco.MjModel, names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return lower and upper limits for scalar joints.

    Unlimited joints use ``[-pi, pi]``.

    Args:
        model: Compiled model.
        names: Joint names.

    Returns:
        ``(lower, upper)`` arrays.
    """
    lower: list[float] = []
    upper: list[float] = []
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id >= 0 and int(model.jnt_limited[joint_id]):
            low, high = model.jnt_range[joint_id]
            lower.append(float(low) + 1e-4)
            upper.append(float(high) - 1e-4)
        else:
            lower.append(-np.pi)
            upper.append(np.pi)
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def _spec(world: SimWorld, instance: str) -> RobotSpec:
    placement = next((item for item in world.scene.robots if item.instance == instance), None)
    if placement is None:
        raise ConfigError("Robot instance is not in the scene", details={"instance": instance})
    return placed_spec(world.catalog, placement)


def _dof_width(joint_type: int) -> int:
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 6
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 3
    return 1
