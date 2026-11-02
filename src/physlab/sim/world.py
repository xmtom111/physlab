"""Owns one compiled MuJoCo scene and the live ``MjData``.

Public methods take a reentrant lock. ``copy_data`` returns an independent
``MjData`` so a planner can read state without writing the live buffer.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from typing import Any

import mujoco
import numpy as np

from physlab.core.config import PhysLabConfig
from physlab.core.errors import AssetMissing, ConfigError
from physlab.core.logging import get_logger
from physlab.core.types import Pose
from physlab.hal.interfaces import Contact
from physlab.robots.catalog import RobotCatalog
from physlab.sim.scene import ObjectSpec, SceneSpec, _add_object, build_mjspec, placed_spec

_logger = get_logger(__name__)


class SimWorld:
    """Steppable MuJoCo world built from a :class:`SceneSpec`."""

    def __init__(self, scene: SceneSpec, catalog: RobotCatalog, config: PhysLabConfig) -> None:
        """Compile ``scene`` and allocate data.

        Args:
            scene: Robots, objects, and cameras.
            catalog: Robot descriptions and asset root.
            config: Timestep, run directory, and the optional viewer flag.

        Raises:
            AssetMissing: The spec does not compile.
            ConfigError: The scene names are inconsistent.
        """
        self.scene = scene
        self.catalog = catalog
        self.config = config
        self._lock = threading.RLock()
        self._spec = build_mjspec(scene, catalog)
        _set_timestep(self._spec, config.sim.timestep_s)
        self.model = _compile(self._spec)
        self.data = mujoco.MjData(self.model)
        self._rng = np.random.default_rng(config.seed)
        self._welds: dict[tuple[str, str], str] = {}
        self._weld_offsets: dict[tuple[str, str], tuple[str, str, Pose]] = {}
        self._kinematic_holds: list[Callable[[], None]] = []
        self._apply_initial_state()
        self._saved = _snapshot(self.model, self.data)
        self._viewer = _maybe_viewer(self)

    def step(self, n: int = 1) -> None:
        """Advance the simulation ``n`` steps.

        Args:
            n: Number of integrator steps. Each step lasts ``timestep_s``.

        Raises:
            ConfigError: ``n`` is negative.
        """
        if n < 0:
            raise ConfigError("step count must be non-negative", details={"n": n})
        with self._lock:
            for _ in range(n):
                mujoco.mj_step(self.model, self.data)
                self._restore_holds()
            self._sync_viewer()

    def run_until(self, predicate: Callable[[SimWorld], bool], timeout_s: float) -> bool:
        """Step until ``predicate`` is true or ``timeout_s`` elapses.

        Args:
            predicate: Called with this world after each check.
            timeout_s: Maximum additional simulated time, seconds.

        Returns:
            True when the predicate became true.
        """
        with self._lock:
            start = float(self.data.time)
            if predicate(self):
                return True
            while float(self.data.time) - start < timeout_s:
                mujoco.mj_step(self.model, self.data)
                self._sync_viewer()
                if predicate(self):
                    return True
            return False

    def reset(self, seed: int | None = None) -> None:
        """Restore the initial joint state and replace the random generator.

        Args:
            seed: Seed for the world's numpy Generator. The previous generator
                is kept when ``seed`` is omitted.
        """
        with self._lock:
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            mujoco.mj_resetData(self.model, self.data)
            _restore(self.model, self.data, self._saved)
            mujoco.mj_forward(self.model, self.data)
            for hold in self._kinematic_holds:
                recapture = getattr(hold, "recapture", None)
                if recapture is not None:
                    recapture()

    @property
    def time_s(self) -> float:
        """Simulation time, seconds."""
        with self._lock:
            return float(self.data.time)

    def add_object_at_runtime(self, obj: ObjectSpec) -> None:
        """Insert ``obj`` and keep the qpos and qvel of joints that already exist.

        Args:
            obj: Object to add. Its name must not already be a body.

        Raises:
            ConfigError: A body with that name already exists.
            AssetMissing: The updated spec does not compile.
        """
        with self._lock:
            if _named(self.model, mujoco.mjtObj.mjOBJ_BODY, obj.name) >= 0:
                raise ConfigError("Object name already exists", details={"name": obj.name})
            saved = _snapshot(self.model, self.data)
            _add_object(self._spec, obj)
            self._recompile(saved)

    def weld(self, body_a: str, body_b: str) -> None:
        """Weld two bodies together with an equality constraint.

        Args:
            body_a: First body name.
            body_b: Second body name.

        Raises:
            ConfigError: A body is missing or the pair is already welded.
            AssetMissing: Recompilation failed.
        """
        with self._lock:
            pair = _weld_key(body_a, body_b)
            if pair in self._welds:
                raise ConfigError("Bodies are already welded", details={"bodies": pair})
            self._require_body(body_a)
            self._require_body(body_b)
            saved = _snapshot(self.model, self.data)
            _sync_world_pose(self._spec, self.model, self.data, body_a)
            _sync_world_pose(self._spec, self.model, self.data, body_b)
            name = f"weld_{len(self._welds)}_{body_a}_{body_b}"
            # A zero data vector tells the compiler to freeze the relative pose
            # from the spec. A hand-filled relpose is reshuffled and pulls the
            # part back toward where it was spawned.
            self._spec.add_equality(
                name=name,
                type=mujoco.mjtEq.mjEQ_WELD,
                name1=body_a,
                name2=body_b,
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                data=[0.0] * int(mujoco.mjNEQDATA),
                active=1,
            )
            self._welds[pair] = name
            offset = _movable_offset(self, body_a, body_b)
            self._recompile(saved)
            if offset is not None:
                self._weld_offsets[pair] = offset

    def unweld(self, body_a: str, body_b: str) -> None:
        """Remove a weld created by :meth:`weld`.

        Args:
            body_a: First body name.
            body_b: Second body name.

        Raises:
            ConfigError: That pair is not welded.
            AssetMissing: Recompilation failed.
        """
        with self._lock:
            pair = _weld_key(body_a, body_b)
            name = self._welds.pop(pair, None)
            self._weld_offsets.pop(pair, None)
            if name is None:
                raise ConfigError("Bodies are not welded", details={"bodies": pair})
            saved = _snapshot(self.model, self.data)
            equality = self._spec.equality(name)
            if equality is not None:
                self._spec.delete(equality)
            self._recompile(saved)

    def project_welds(self) -> None:
        """Place each free welded body on the pose captured at :meth:`weld`.

        The focus screw is a position joint. A welded part follows it only
        after the constraint is integrated, so alignment sets the pose directly.
        """
        with self._lock:
            for driver, follower, relative in self._weld_offsets.values():
                if max(abs(value) for value in relative.position_m) > 0.05:
                    continue
                _write_free_pose(self, follower, self.body_pose(driver).compose(relative))
            mujoco.mj_forward(self.model, self.data)

    def contacts(self) -> list[Contact]:
        """Return the active contacts.

        Returns:
            One entry per MuJoCo contact, with geom and body names.
        """
        with self._lock:
            found: list[Contact] = []
            for index in range(self.data.ncon):
                contact = self.data.contact[index]
                geom1 = int(contact.geom1)
                geom2 = int(contact.geom2)
                body1 = int(self.model.geom_bodyid[geom1])
                body2 = int(self.model.geom_bodyid[geom2])
                found.append(
                    Contact(
                        geom1=_name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
                        geom2=_name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
                        body1=_name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1),
                        body2=_name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2),
                        dist=float(contact.dist),
                    )
                )
            return found

    def copy_data(self) -> mujoco.MjData:
        """Return a forward-kinematics copy of the live data.

        The returned buffer is not the live ``MjData``. Planners should read
        it and leave the simulation data alone.

        Returns:
            Independent ``MjData`` with the current qpos, qvel, and ctrl.
        """
        with self._lock:
            cloned = mujoco.MjData(self.model)
            cloned.qpos[:] = self.data.qpos
            cloned.qvel[:] = self.data.qvel
            cloned.ctrl[:] = self.data.ctrl
            cloned.time = self.data.time
            mujoco.mj_forward(self.model, cloned)
            return cloned

    def joint_position(self, name: str) -> float:
        """Return one hinge or slide joint position.

        Args:
            name: Joint name.

        Returns:
            Position in radians or meters.

        Raises:
            ConfigError: The joint is missing or is not a scalar joint.
        """
        with self._lock:
            joint_id = self._require_joint(name)
            if int(self.model.jnt_type[joint_id]) not in {
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            }:
                raise ConfigError("Expected a hinge or slide joint", details={"joint": name})
            address = int(self.model.jnt_qposadr[joint_id])
            return float(self.data.qpos[address])

    def joint_velocity(self, name: str) -> float:
        """Return one hinge or slide joint velocity.

        Args:
            name: Joint name.

        Returns:
            Velocity in radians per second or meters per second.
        """
        with self._lock:
            joint_id = self._require_joint(name)
            address = int(self.model.jnt_dofadr[joint_id])
            return float(self.data.qvel[address])

    def set_joint_position(self, name: str, value: float) -> None:
        """Write one scalar joint position and run forward kinematics.

        Args:
            name: Joint name.
            value: Position in radians or meters.
        """
        with self._lock:
            joint_id = self._require_joint(name)
            address = int(self.model.jnt_qposadr[joint_id])
            self.data.qpos[address] = value
            mujoco.mj_forward(self.model, self.data)

    def actuator_ctrl(self, name: str) -> float:
        """Return one actuator command.

        Args:
            name: Actuator name.

        Returns:
            Current ``ctrl`` value.
        """
        with self._lock:
            return float(self.data.ctrl[self._require_actuator(name)])

    def set_actuator_ctrl(self, name: str, value: float) -> None:
        """Write one actuator command.

        Args:
            name: Actuator name.
            value: Command in the actuator's units.
        """
        with self._lock:
            self.data.ctrl[self._require_actuator(name)] = value

    def site_pose(self, name: str) -> Pose:
        """Return a site pose in the world frame.

        Args:
            name: Site name.

        Returns:
            Pose expressed in ``world``.
        """
        with self._lock:
            site_id = _named(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id < 0:
                raise ConfigError("Site was not found", details={"site": name})
            return _pose_from_xmat(self.data.site_xpos[site_id], self.data.site_xmat[site_id])

    def body_pose(self, name: str) -> Pose:
        """Return a body pose in the world frame.

        Args:
            name: Body name.

        Returns:
            Pose expressed in ``world``.
        """
        with self._lock:
            body_id = self._require_body(name)
            return _pose_from_xmat(self.data.xpos[body_id], self.data.xmat[body_id])

    def _apply_initial_state(self) -> None:
        for placement in self.scene.robots:
            robot = placed_spec(self.catalog, placement)
            pairs = zip(robot.sim.arm_joints, robot.sim.home_qpos_rad, strict=False)
            for joint_name, value in pairs:
                full = f"{placement.instance}/{joint_name}"
                joint_id = _named(self.model, mujoco.mjtObj.mjOBJ_JOINT, full)
                if joint_id < 0:
                    continue
                address = int(self.model.jnt_qposadr[joint_id])
                self.data.qpos[address] = value
            for index, actuator_name in enumerate(robot.sim.arm_actuators):
                if index >= len(robot.sim.home_qpos_rad):
                    break
                actuator = f"{placement.instance}/{actuator_name}"
                actuator_id = _named(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
                if actuator_id >= 0:
                    self.data.ctrl[actuator_id] = robot.sim.home_qpos_rad[index]
        mujoco.mj_forward(self.model, self.data)

    def _recompile(self, saved: _State) -> None:
        self.model = _compile(self._spec)
        self.data = mujoco.MjData(self.model)
        _restore(self.model, self.data, saved)
        mujoco.mj_forward(self.model, self.data)
        self._sync_viewer()

    def _require_body(self, name: str) -> int:
        body_id = _named(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ConfigError("Body was not found", details={"body": name})
        return body_id

    def _require_joint(self, name: str) -> int:
        joint_id = _named(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ConfigError("Joint was not found", details={"joint": name})
        return joint_id

    def _require_actuator(self, name: str) -> int:
        actuator_id = _named(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise ConfigError("Actuator was not found", details={"actuator": name})
        return actuator_id

    def _restore_holds(self) -> None:
        for hold in self._kinematic_holds:
            hold()

    def _sync_viewer(self) -> None:
        if self._viewer is not None:
            self._viewer.sync()


class _State:
    def __init__(self) -> None:
        self.qpos: dict[str, np.ndarray] = {}
        self.qvel: dict[str, np.ndarray] = {}
        self.ctrl: dict[str, float] = {}


def _snapshot(model: mujoco.MjModel, data: mujoco.MjData) -> _State:
    state = _State()
    for joint_id in range(model.njnt):
        name = _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not name:
            continue
        q_address = int(model.jnt_qposadr[joint_id])
        v_address = int(model.jnt_dofadr[joint_id])
        q_count = _qpos_width(int(model.jnt_type[joint_id]))
        v_count = _dof_width(int(model.jnt_type[joint_id]))
        state.qpos[name] = np.array(data.qpos[q_address : q_address + q_count], dtype=np.float64)
        state.qvel[name] = np.array(data.qvel[v_address : v_address + v_count], dtype=np.float64)
    for actuator_id in range(model.nu):
        name = _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if name:
            state.ctrl[name] = float(data.ctrl[actuator_id])
    return state


def _restore(model: mujoco.MjModel, data: mujoco.MjData, state: _State) -> None:
    for joint_id in range(model.njnt):
        name = _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name not in state.qpos:
            continue
        q_address = int(model.jnt_qposadr[joint_id])
        v_address = int(model.jnt_dofadr[joint_id])
        q_values = state.qpos[name]
        v_values = state.qvel[name]
        data.qpos[q_address : q_address + q_values.size] = q_values
        data.qvel[v_address : v_address + v_values.size] = v_values
    for actuator_id in range(model.nu):
        name = _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if name in state.ctrl:
            data.ctrl[actuator_id] = state.ctrl[name]


def _compile(spec: mujoco.MjSpec) -> mujoco.MjModel:
    try:
        return spec.compile()
    except (OSError, ValueError) as exc:
        raise AssetMissing("Scene failed to compile", details={"reason": str(exc)}) from exc


def _set_timestep(spec: mujoco.MjSpec, timestep_s: float) -> None:
    spec.option.timestep = timestep_s


def _maybe_viewer(world: SimWorld) -> Any:
    if not world.config.sim.viewer or not os.environ.get("DISPLAY"):
        return None
    try:
        import mujoco.viewer
    except ImportError as exc:
        _logger.info("viewer_unavailable", reason=str(exc))
        return None
    return mujoco.viewer.launch_passive(world.model, world.data)


def _sync_world_pose(
    spec: mujoco.MjSpec,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    name: str,
) -> None:
    """Write a world-child's live pose into the spec before a weld compiles.

    The compiler freezes the weld's relative pose from the spec, not from
    ``qpos``. A part that has moved since spawn would otherwise be pulled
    back to the pickup slot.
    """
    body = spec.body(name)
    if body is None or body.parent is None or body.parent.name != "world":
        return
    pose = _pose_from_xmat(
        data.xpos[_named(model, mujoco.mjtObj.mjOBJ_BODY, name)],
        data.xmat[_named(model, mujoco.mjtObj.mjOBJ_BODY, name)],
    )
    body.pos = list(pose.position_m)
    body.quat = list(pose.quat_wxyz)


def _movable_offset(world: SimWorld, body_a: str, body_b: str) -> tuple[str, str, Pose] | None:
    free_a = _free_joint(world, body_a)
    free_b = _free_joint(world, body_b)
    if free_a >= 0 and free_b < 0:
        driver, follower = body_b, body_a
    elif free_b >= 0 and free_a < 0:
        driver, follower = body_a, body_b
    else:
        return None
    follower_pose = world.body_pose(follower).model_copy(update={"frame": "body"})
    relative = world.body_pose(driver).inverse().compose(follower_pose)
    return driver, follower, relative


def _free_joint(world: SimWorld, body: str) -> int:
    return int(mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_JOINT, f"{body}_free"))


def _write_free_pose(world: SimWorld, body: str, pose: Pose) -> None:
    joint_id = _free_joint(world, body)
    if joint_id < 0:
        return
    address = int(world.model.jnt_qposadr[joint_id])
    world.data.qpos[address : address + 3] = pose.position_m
    world.data.qpos[address + 3 : address + 7] = pose.quat_wxyz
    velocity = int(world.model.jnt_dofadr[joint_id])
    world.data.qvel[velocity : velocity + 6] = 0.0


def _weld_key(body_a: str, body_b: str) -> tuple[str, str]:
    return (body_a, body_b) if body_a <= body_b else (body_b, body_a)


def _pose_from_xmat(position: np.ndarray, xmat: np.ndarray) -> Pose:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return Pose.from_matrix(transform, frame="world")


def _named(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    return int(mujoco.mj_name2id(model, kind, name))


def _name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    found = mujoco.mj_id2name(model, kind, index)
    return "" if found is None else str(found)


def _qpos_width(joint_type: int) -> int:
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4
    return 1


def _dof_width(joint_type: int) -> int:
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 6
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 3
    return 1
