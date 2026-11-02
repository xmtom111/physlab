"""MuJoCo implementations of the arm, gripper, base, and camera protocols."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import mujoco
import numpy as np

from physlab.core.errors import ConfigError
from physlab.core.types import FailureReason, JointState, MotionResult, Pose, Pose2D
from physlab.hal.interfaces import CameraIntrinsics, Frame
from physlab.robots.compose import RUNTIME_EE_SITE
from physlab.robots.models import RobotSpec
from physlab.sim.gl import renderer_class
from physlab.sim.world import SimWorld

_SETTLE_RAD = 2e-3
_SETTLE_VEL = 1e-2
_DEFAULT_VEL = 1.0
_CONTACT_DIST = 1e-4


class SimArm:
    """Joint-space arm driven by actuator setpoints."""

    def __init__(self, world: SimWorld, instance: str, spec: RobotSpec) -> None:
        """Bind an arm instance inside ``world``.

        Args:
            world: Simulation that owns the model.
            instance: Prefix used when the robot was placed.
            spec: Catalog entry for this robot.
        """
        self.name = instance
        self._world = world
        self._spec = spec
        self.joint_names = tuple(f"{instance}/{name}" for name in spec.sim.arm_joints)
        self.dof = len(self.joint_names)
        actuators = spec.sim.arm_actuators
        self._actuators = tuple(f"{instance}/{name}" for name in actuators)
        if spec.sim.ee_site:
            self._ee_site = f"{instance}/{spec.sim.ee_site}"
        else:
            self._ee_site = f"{instance}/{RUNTIME_EE_SITE}"
        self._limits = _velocity_limits(spec)

    def get_joint_state(self) -> JointState:
        """Return arm joint positions and velocities.

        Returns:
            Joint state in ``joint_names`` order.
        """
        positions = tuple(self._world.joint_position(name) for name in self.joint_names)
        velocities = tuple(self._world.joint_velocity(name) for name in self.joint_names)
        return JointState(
            names=self.joint_names, positions_rad=positions, velocities_rad_s=velocities
        )

    def get_ee_pose(self) -> Pose:
        """Return the end-effector site pose in the world frame.

        Returns:
            Site pose.
        """
        return self._world.site_pose(self._ee_site)

    def move_joints(
        self,
        q_rad: Sequence[float],
        *,
        speed_scale: float = 0.5,
        timeout_s: float = 10.0,
    ) -> MotionResult:
        """Follow a minimum-jerk joint path and hold the target until settled.

        Args:
            q_rad: Target positions, one per arm joint.
            speed_scale: Fraction of the per-joint velocity limit.
            timeout_s: Maximum simulated time, seconds.

        Returns:
            Success, collision, timeout, or a joint-limit failure.

        Raises:
            ConfigError: The target length or ``speed_scale`` is invalid.
        """
        if speed_scale <= 0:
            raise ConfigError("speed_scale must be positive", details={"speed_scale": speed_scale})
        target = [float(value) for value in q_rad]
        if len(target) != self.dof:
            raise ConfigError(
                "Target length must match the arm dof",
                details={"dof": self.dof, "length": len(target)},
            )
        limit_error = _outside_limits(self._world, self.joint_names, target)
        if limit_error is not None:
            return _failed(FailureReason.JOINT_LIMIT, limit_error, 0.0)
        start = [self._world.joint_position(name) for name in self.joint_names]
        delta = [goal - current for goal, current in zip(target, start, strict=True)]
        duration = _path_duration(delta, self._limits, speed_scale)
        return self._follow(start, target, duration, timeout_s)

    def move_pose(
        self,
        target: Pose,
        *,
        speed_scale: float = 0.5,
        timeout_s: float = 10.0,
    ) -> MotionResult:
        """Pose goals are implemented with inverse kinematics in T4.

        Args:
            target: Desired end-effector pose. Unused.
            speed_scale: Unused.
            timeout_s: Unused.

        Raises:
            NotImplementedError: Always. Implemented in T4.
        """
        del target, speed_scale, timeout_s
        raise NotImplementedError("Implemented in T4")

    def stop(self) -> None:
        """Hold the current joint position, including a gravity trim."""
        current = [self._world.joint_position(name) for name in self.joint_names]
        self._write_ctrl(current)

    def _follow(
        self,
        start: list[float],
        target: list[float],
        duration_s: float,
        timeout_s: float,
    ) -> MotionResult:
        world = self._world
        t0 = world.time_s
        initial = _external_pairs(world, self.name)
        while True:
            elapsed = world.time_s - t0
            if elapsed > timeout_s:
                return _failed(FailureReason.TIMEOUT, "joint motion timed out", elapsed, self)
            fraction = 1.0 if duration_s <= 0 else min(elapsed / duration_s, 1.0)
            scale = _minimum_jerk(fraction)
            command = [a + (b - a) * scale for a, b in zip(start, target, strict=True)]
            self._write_ctrl(command)
            world.step()
            hit = _new_collision(world, self.name, self._finger_names(), initial)
            if hit is not None:
                return _failed(FailureReason.COLLISION, hit, world.time_s - t0, self)
            if fraction >= 1.0 and _settled(self, target):
                return _ok(world.time_s - t0, self, target)

    def _write_ctrl(self, command: list[float]) -> None:
        model = self._world.model
        bias = self._world.data.qfrc_bias
        for value, actuator, joint in zip(command, self._actuators, self.joint_names, strict=False):
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            trim = 0.0
            if actuator_id >= 0 and joint_id >= 0:
                stiffness = float(model.actuator_gainprm[actuator_id][0])
                if stiffness > 1e-6:
                    dof = int(model.jnt_dofadr[joint_id])
                    trim = float(bias[dof]) / stiffness
            self._world.set_actuator_ctrl(actuator, value + trim)

    def _finger_names(self) -> set[str]:
        gripper = self._spec.sim.gripper
        if gripper is None:
            return set()
        prefix = f"{self.name}/"
        attach = gripper.attach_from
        extra = f"{attach}/" if attach else ""
        return {f"{prefix}{extra}{name}" for name in gripper.finger_geoms}


class SimGripper:
    """Gripper commanded by interpolating between open and closed setpoints."""

    def __init__(self, world: SimWorld, instance: str, spec: RobotSpec) -> None:
        """Bind the gripper actuator for ``instance``.

        Args:
            world: Simulation world.
            instance: Robot instance prefix.
            spec: Catalog entry. ``sim.gripper`` must be set.
        """
        if spec.sim.gripper is None:
            raise ConfigError("Robot has no gripper", details={"robot": spec.id})
        self._world = world
        self._spec = spec
        self._instance = instance
        attach = spec.sim.gripper.attach_from
        extra = f"{attach}/" if attach else ""
        self._actuator = f"{instance}/{extra}{spec.sim.gripper.actuator}"
        self._open = spec.sim.gripper.open_ctrl
        self._closed = spec.sim.gripper.closed_ctrl
        self._fingers = tuple(f"{instance}/{extra}{name}" for name in spec.sim.gripper.finger_geoms)

    def open(self) -> MotionResult:
        """Move the gripper command to ``open_ctrl``.

        Returns:
            Motion result for the interpolation.
        """
        return self._goto(self._open)

    def close(self, force_scale: float = 0.5) -> MotionResult:
        """Move the gripper command to ``closed_ctrl``.

        Args:
            force_scale: Accepted for the real-hardware interface. The
                simulation uses the catalog closed command.

        Returns:
            Motion result for the interpolation.
        """
        del force_scale
        return self._goto(self._closed)

    def get_opening(self) -> float:
        """Return 1 at ``open_ctrl`` and 0 at ``closed_ctrl``.

        Returns:
            Opening fraction clipped to ``[0, 1]``.
        """
        span = self._open - self._closed
        if abs(span) < 1e-9:
            return 0.0
        raw = (self._world.actuator_ctrl(self._actuator) - self._closed) / span
        return float(min(max(raw, 0.0), 1.0))

    def is_grasping(self) -> bool:
        """Return whether both finger sides touch one external body.

        Returns:
            True when a non-robot body is in contact with a left and a right
            finger geom.
        """
        left = {name for name in self._fingers if "left" in name}
        right = {name for name in self._fingers if "right" in name}
        if not left or not right:
            return False
        touched_left: set[str] = set()
        touched_right: set[str] = set()
        prefix = f"{self._instance}/"
        for contact in self._world.contacts():
            if contact.dist > _CONTACT_DIST:
                continue
            other = _external_body(contact.body1, contact.body2, prefix)
            if other is None:
                continue
            if contact.geom1 in left or contact.geom2 in left:
                touched_left.add(other)
            if contact.geom1 in right or contact.geom2 in right:
                touched_right.add(other)
        return bool(touched_left & touched_right)

    def _goto(self, target: float) -> MotionResult:
        start = self._world.actuator_ctrl(self._actuator)
        t0 = self._world.time_s
        duration = 0.4
        while True:
            elapsed = self._world.time_s - t0
            fraction = min(elapsed / duration, 1.0)
            value = start + (target - start) * _minimum_jerk(fraction)
            self._world.set_actuator_ctrl(self._actuator, value)
            self._world.step()
            if fraction >= 1.0:
                return MotionResult(
                    success=True,
                    reason=None,
                    final_pos_error_m=0.0,
                    final_rot_error_rad=abs(target - value),
                    duration_s=elapsed,
                )


class SimMobileBase:
    """Differential-drive or holonomic base."""

    def __init__(self, world: SimWorld, instance: str, spec: RobotSpec) -> None:
        """Bind base joints for ``instance``.

        Args:
            world: Simulation world.
            instance: Robot instance prefix.
            spec: Catalog entry with base joints.
        """
        self._world = world
        self._instance = instance
        self._spec = spec
        self._joints = tuple(f"{instance}/{name}" for name in spec.sim.base_joints)
        self._actuators = tuple(f"{instance}/{name}" for name in spec.sim.base_actuators)
        self._free_joint = f"{instance}_base_free"
        if spec.sim.base_type.value == "diff_drive":
            # kv on the wheel actuators makes the contact solver diverge. The
            # base pose is integrated on the anchor free joint instead, and the
            # wheel commands are still written to ctrl.
            _zero_actuator_gain(world, self._actuators)

    def get_pose2d(self) -> Pose2D:
        """Return the chassis pose projected to the plane.

        Returns:
            Pose2D in the world frame.
        """
        pose = self._world.body_pose(self._body_name())
        matrix = pose.to_matrix()
        yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
        return Pose2D(x_m=pose.position_m[0], y_m=pose.position_m[1], yaw_rad=yaw, frame="world")

    def set_velocity(
        self,
        vx_m_s: float,
        vy_m_s: float,
        wz_rad_s: float,
        duration_s: float,
    ) -> MotionResult:
        """Hold a planar velocity, then stop.

        Args:
            vx_m_s: Forward speed, meters per second.
            vy_m_s: Lateral speed. Used for holonomic joints only.
            wz_rad_s: Yaw rate, radians per second.
            duration_s: How long to apply the command, seconds.

        Returns:
            Success after the duration elapses.
        """
        if self._spec.sim.base_type.value == "holonomic_joints":
            return self._holonomic_velocity(vx_m_s, vy_m_s, wz_rad_s, duration_s)
        return self._diffdrive_velocity(vx_m_s, wz_rad_s, duration_s)

    def move_to(
        self,
        target: Pose2D,
        *,
        pos_tol_m: float = 0.02,
        yaw_tol_rad: float = 0.05,
        timeout_s: float = 30.0,
    ) -> MotionResult:
        """Drive to ``target``.

        Differential drive rotates, drives straight, then rotates again.
        Holonomic bases servo the base joints to the target coordinates.

        Args:
            target: Goal pose in the world frame.
            pos_tol_m: Position tolerance, meters.
            yaw_tol_rad: Yaw tolerance, radians.
            timeout_s: Maximum simulated time, seconds.

        Returns:
            Success or timeout.
        """
        if self._spec.sim.base_type.value == "holonomic_joints":
            return self._holonomic_move(target, pos_tol_m, yaw_tol_rad, timeout_s)
        return self._diffdrive_move(target, pos_tol_m, yaw_tol_rad, timeout_s)

    def stop(self) -> None:
        """Zero velocity commands, or hold holonomic joints in place."""
        if self._spec.sim.base_type.value == "holonomic_joints":
            for joint, actuator in zip(self._joints, self._actuators, strict=False):
                self._world.set_actuator_ctrl(actuator, self._world.joint_position(joint))
            return
        for actuator in self._actuators:
            self._world.set_actuator_ctrl(actuator, 0.0)

    def _body_name(self) -> str:
        chassis = f"{self._instance}/chassis"
        try:
            self._world.body_pose(chassis)
        except ConfigError:
            return f"{self._instance}_anchor"
        return chassis

    def _diffdrive_velocity(
        self, vx_m_s: float, wz_rad_s: float, duration_s: float
    ) -> MotionResult:
        t0 = self._world.time_s
        while self._world.time_s - t0 < duration_s:
            self._command_twist(vx_m_s, wz_rad_s)
        self.stop()
        return MotionResult(
            success=True,
            reason=None,
            final_pos_error_m=0.0,
            final_rot_error_rad=0.0,
            duration_s=self._world.time_s - t0,
        )

    def _diffdrive_move(
        self,
        target: Pose2D,
        pos_tol_m: float,
        yaw_tol_rad: float,
        timeout_s: float,
    ) -> MotionResult:
        t0 = self._world.time_s
        while self._world.time_s - t0 < timeout_s:
            pose = self.get_pose2d()
            dx = target.x_m - pose.x_m
            dy = target.y_m - pose.y_m
            distance = math.hypot(dx, dy)
            heading = math.atan2(dy, dx)
            yaw_error = _wrap(target.yaw_rad - pose.yaw_rad)
            # Stop inside the tolerance so the last step does not land on the bound.
            if distance <= pos_tol_m * 0.5 and abs(yaw_error) <= yaw_tol_rad * 0.5:
                self.stop()
                return MotionResult(
                    success=True,
                    reason=None,
                    final_pos_error_m=distance,
                    final_rot_error_rad=abs(yaw_error),
                    duration_s=self._world.time_s - t0,
                )
            if distance > pos_tol_m * 0.5:
                face = _wrap(heading - pose.yaw_rad)
                if abs(face) > 0.15:
                    vx, wz = 0.0, _clip(2.5 * face, 1.5)
                else:
                    vx, wz = _clip(0.8 * distance, 0.5), _clip(2.0 * face, 0.8)
            else:
                vx, wz = 0.0, _clip(2.5 * yaw_error, 1.5)
            self._command_twist(vx, wz)
        self.stop()
        pose = self.get_pose2d()
        distance = math.hypot(target.x_m - pose.x_m, target.y_m - pose.y_m)
        yaw_error = abs(_wrap(target.yaw_rad - pose.yaw_rad))
        return MotionResult(
            success=False,
            reason=FailureReason.TIMEOUT,
            message="base move_to timed out",
            final_pos_error_m=distance,
            final_rot_error_rad=yaw_error,
            duration_s=self._world.time_s - t0,
        )

    def _wheel_ctrl(self, vx_m_s: float, wz_rad_s: float) -> tuple[float, float]:
        radius, half_track = self._wheel_geometry()
        left = (vx_m_s - wz_rad_s * half_track) / radius
        right = (vx_m_s + wz_rad_s * half_track) / radius
        return left, right

    def _command_twist(self, vx_m_s: float, wz_rad_s: float) -> None:
        left, right = self._wheel_ctrl(vx_m_s, wz_rad_s)
        self._set_wheels(left, right)
        pose = self.get_pose2d()
        dt = float(self._world.model.opt.timestep)
        cosine, sine = math.cos(pose.yaw_rad), math.sin(pose.yaw_rad)
        x_m = pose.x_m + vx_m_s * cosine * dt
        y_m = pose.y_m + vx_m_s * sine * dt
        yaw = pose.yaw_rad + wz_rad_s * dt
        self._write_planar(x_m, y_m, yaw)
        self._spin_wheel(self._joints[0], left * dt)
        self._spin_wheel(self._joints[1], right * dt)
        self._world.step()
        self._write_planar(x_m, y_m, yaw)

    def _write_planar(self, x_m: float, y_m: float, yaw_rad: float) -> None:
        model = self._world.model
        data = self._world.data
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self._free_joint)
        if joint_id < 0:
            raise ConfigError(
                "Differential drive anchor has no free joint",
                details={"joint": self._free_joint},
            )
        q_address = int(model.jnt_qposadr[joint_id])
        v_address = int(model.jnt_dofadr[joint_id])
        height = float(data.qpos[q_address + 2])
        half = 0.5 * yaw_rad
        data.qpos[q_address : q_address + 7] = [
            x_m,
            y_m,
            height,
            math.cos(half),
            0.0,
            0.0,
            math.sin(half),
        ]
        data.qvel[v_address : v_address + 6] = 0.0
        mujoco.mj_forward(model, data)

    def _spin_wheel(self, joint: str, delta_rad: float) -> None:
        self._world.set_joint_position(joint, self._world.joint_position(joint) + delta_rad)

    def _set_wheels(self, left: float, right: float) -> None:
        if len(self._actuators) < 2:
            raise ConfigError("Differential drive needs two wheel actuators")
        self._world.set_actuator_ctrl(self._actuators[0], left)
        self._world.set_actuator_ctrl(self._actuators[1], right)

    def _wheel_geometry(self) -> tuple[float, float]:
        radius = 0.05
        geom = f"{self._instance}/left_wheel_geom"
        geom_id = mujoco.mj_name2id(self._world.model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        if geom_id >= 0:
            radius = float(self._world.model.geom_size[geom_id][0])
        try:
            left = self._world.body_pose(f"{self._instance}/left_wheel")
            right = self._world.body_pose(f"{self._instance}/right_wheel")
        except ConfigError:
            return radius, 0.15
        track = math.hypot(
            left.position_m[0] - right.position_m[0], left.position_m[1] - right.position_m[1]
        )
        return radius, max(track / 2.0, 1e-3)

    def _holonomic_velocity(
        self,
        vx_m_s: float,
        vy_m_s: float,
        wz_rad_s: float,
        duration_s: float,
    ) -> MotionResult:
        current = [self._world.joint_position(name) for name in self._joints]
        rates = [vx_m_s, vy_m_s, wz_rad_s]
        target = [value + rates[index] * duration_s for index, value in enumerate(current)]
        return self._servo(target, max(duration_s, 0.05), max(duration_s + 2.0, 2.0))

    def _holonomic_move(
        self,
        target: Pose2D,
        pos_tol_m: float,
        yaw_tol_rad: float,
        timeout_s: float,
    ) -> MotionResult:
        del pos_tol_m, yaw_tol_rad
        goal = [target.x_m, target.y_m, target.yaw_rad]
        return self._servo(goal[: len(self._joints)], 1.0, timeout_s)

    def _servo(self, target: list[float], duration_s: float, timeout_s: float) -> MotionResult:
        start = [self._world.joint_position(name) for name in self._joints]
        t0 = self._world.time_s
        while self._world.time_s - t0 < timeout_s:
            elapsed = self._world.time_s - t0
            fraction = min(elapsed / duration_s, 1.0) if duration_s > 0 else 1.0
            scale = _minimum_jerk(fraction)
            for index, actuator in enumerate(self._actuators):
                value = start[index] + (target[index] - start[index]) * scale
                self._world.set_actuator_ctrl(actuator, value)
            self._world.step()
            if fraction >= 1.0:
                error = max(
                    abs(self._world.joint_position(name) - target[index])
                    for index, name in enumerate(self._joints)
                )
                if error < 0.02:
                    return MotionResult(
                        success=True,
                        reason=None,
                        final_pos_error_m=error,
                        final_rot_error_rad=0.0,
                        duration_s=self._world.time_s - t0,
                    )
        return _failed(FailureReason.TIMEOUT, "holonomic base timed out", self._world.time_s - t0)


class SimCamera:
    """Offscreen RGB and depth camera."""

    def __init__(
        self,
        world: SimWorld,
        name: str,
        *,
        width: int,
        height: int,
        fovy_deg: float,
    ) -> None:
        """Create a renderer aimed at ``name``.

        Args:
            world: Simulation world.
            name: MuJoCo camera name.
            width: Image width, pixels.
            height: Image height, pixels.
            fovy_deg: Vertical field of view, degrees.
        """
        self.name = name
        self._world = world
        self._width = width
        self._height = height
        self._fovy_deg = fovy_deg
        self._renderer: Any = None
        self._renderer_model: mujoco.MjModel | None = None
        self._intrinsics = _intrinsics(width, height, fovy_deg)

    def _ensure_renderer(self) -> None:
        if self._renderer is not None and self._renderer_model is self._world.model:
            return
        renderer_type = renderer_class()
        self._renderer = renderer_type(self._world.model, self._height, self._width)
        self._renderer_model = self._world.model

    def capture(self) -> Frame:
        """Render RGB and depth for the current state.

        Returns:
            Frame with depth in meters.
        """
        world = self._world
        with world._lock:
            self._ensure_renderer()
            self._renderer.update_scene(world.data, camera=self.name)
            rgb = np.asarray(self._renderer.render(), dtype=np.uint8).copy()
            self._renderer.enable_depth_rendering()
            self._renderer.update_scene(world.data, camera=self.name)
            depth = np.asarray(self._renderer.render(), dtype=np.float32).copy()
            self._renderer.disable_depth_rendering()
            camera_pose = _camera_pose(world, self.name)
            timestamp = float(world.data.time)
        return Frame(
            rgb=rgb,
            depth_m=depth,
            intrinsics=self._intrinsics,
            pose_in_world=camera_pose,
            timestamp_s=timestamp,
        )


def _camera_pose(world: SimWorld, name: str) -> Pose | None:
    camera_id = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if camera_id < 0:
        return None
    return _pose_of(world.data.cam_xpos[camera_id], world.data.cam_xmat[camera_id])


def _pose_of(position: np.ndarray, xmat: np.ndarray) -> Pose:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return Pose.from_matrix(transform, frame="world")


def _intrinsics(width: int, height: int, fovy_deg: float) -> CameraIntrinsics:
    fy = 0.5 * height / math.tan(math.radians(fovy_deg) / 2.0)
    return CameraIntrinsics(
        fx=fy, fy=fy, cx=width / 2.0, cy=height / 2.0, width=width, height=height
    )


def _velocity_limits(spec: RobotSpec) -> list[float]:
    raw = spec.sim.joint_velocity_max_rad_s
    limits: list[float] = []
    for index in range(len(spec.sim.arm_joints)):
        if index < len(raw) and raw[index] > 0:
            limits.append(float(raw[index]))
        else:
            limits.append(_DEFAULT_VEL)
    return limits


def _path_duration(delta: list[float], limits: list[float], speed_scale: float) -> float:
    times = [
        abs(value) / max(limit * speed_scale, 1e-4)
        for value, limit in zip(delta, limits, strict=True)
    ]
    return max([*times, 0.05])


def _minimum_jerk(fraction: float) -> float:
    tau = min(max(fraction, 0.0), 1.0)
    return (10.0 * tau**3) - (15.0 * tau**4) + (6.0 * tau**5)


def _settled(arm: SimArm, target: list[float]) -> bool:
    error = max(
        abs(arm._world.joint_position(name) - target[index])
        for index, name in enumerate(arm.joint_names)
    )
    speed = max(abs(arm._world.joint_velocity(name)) for name in arm.joint_names)
    return error < _SETTLE_RAD and speed < _SETTLE_VEL


def _outside_limits(world: SimWorld, names: tuple[str, ...], target: list[float]) -> str | None:
    for name, value in zip(names, target, strict=True):
        joint_id = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0 or not int(world.model.jnt_limited[joint_id]):
            continue
        low, high = world.model.jnt_range[joint_id]
        if value < float(low) - 1e-6 or value > float(high) + 1e-6:
            return f"{name} target {value} is outside [{low}, {high}]"
    return None


def _external_pairs(world: SimWorld, instance: str) -> set[tuple[str, str]]:
    prefix = f"{instance}/"
    pairs: set[tuple[str, str]] = set()
    for contact in world.contacts():
        if contact.dist > _CONTACT_DIST:
            continue
        if _external_body(contact.body1, contact.body2, prefix) is None:
            continue
        pairs.add(_body_pair(contact.body1, contact.body2))
    return pairs


def _new_collision(
    world: SimWorld,
    instance: str,
    fingers: set[str],
    initial: set[tuple[str, str]],
) -> str | None:
    prefix = f"{instance}/"
    grasped = _grasped_bodies(world, prefix, fingers)
    for contact in world.contacts():
        if contact.dist > _CONTACT_DIST:
            continue
        other = _external_body(contact.body1, contact.body2, prefix)
        if other is None or other in grasped:
            continue
        pair = _body_pair(contact.body1, contact.body2)
        if pair in initial:
            continue
        robot_geom = contact.geom1 if contact.body1.startswith(prefix) else contact.geom2
        if robot_geom in fingers:
            continue
        return f"contact {contact.body1} and {contact.body2}"
    return None


def _grasped_bodies(world: SimWorld, prefix: str, fingers: set[str]) -> set[str]:
    left = {name for name in fingers if "left" in name}
    right = {name for name in fingers if "right" in name}
    touched_left: set[str] = set()
    touched_right: set[str] = set()
    for contact in world.contacts():
        other = _external_body(contact.body1, contact.body2, prefix)
        if other is None:
            continue
        if contact.geom1 in left or contact.geom2 in left:
            touched_left.add(other)
        if contact.geom1 in right or contact.geom2 in right:
            touched_right.add(other)
    return touched_left & touched_right


def _zero_actuator_gain(world: SimWorld, actuators: tuple[str, ...]) -> None:
    for name in actuators:
        actuator_id = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            continue
        world.model.actuator_gainprm[actuator_id] = 0.0
        world.model.actuator_biasprm[actuator_id] = 0.0


def _body_pair(body1: str, body2: str) -> tuple[str, str]:
    if body1 <= body2:
        return (body1, body2)
    return (body2, body1)


def _external_body(body1: str, body2: str, prefix: str) -> str | None:
    first = body1.startswith(prefix)
    second = body2.startswith(prefix)
    if first and not second:
        return body2
    if second and not first:
        return body1
    return None


def _ok(duration_s: float, arm: SimArm, target: list[float]) -> MotionResult:
    error = max(
        abs(arm._world.joint_position(name) - target[index])
        for index, name in enumerate(arm.joint_names)
    )
    return MotionResult(
        success=True,
        reason=None,
        final_pos_error_m=0.0,
        final_rot_error_rad=error,
        duration_s=duration_s,
    )


def _failed(
    reason: FailureReason,
    message: str,
    duration_s: float,
    arm: SimArm | None = None,
) -> MotionResult:
    error = 0.0
    if arm is not None and arm.joint_names:
        error = max(abs(arm._world.joint_velocity(name)) for name in arm.joint_names)
    return MotionResult(
        success=False,
        reason=reason,
        message=message,
        final_pos_error_m=0.0,
        final_rot_error_rad=error,
        duration_s=duration_s,
    )


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _clip(value: float, limit: float) -> float:
    return float(min(max(value, -limit), limit))
