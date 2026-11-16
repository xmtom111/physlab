"""One MCP process owns one simulated session.

Real mode still builds the scene so an agent can inspect it, but every tool
that would move hardware or dispense liquid calls :func:`physlab.safety.gate`
and then refuses to drive the simulator as if it were the hardware.
"""

from __future__ import annotations

import base64
import copy
import importlib.util
import threading
from pathlib import Path
from typing import TypeVar, cast

import cv2
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from physlab.assembly.executor import AssemblyExecutor
from physlab.assembly.plan import AssemblyPlan, validate_plan
from physlab.chem.protocol import ChemProtocol, validate_chem_protocol
from physlab.chem.safety import ChemSafetyPolicy
from physlab.controllers.codecheck import check_source
from physlab.controllers.runner import Scenario, run_controller
from physlab.core.config import PhysLabConfig
from physlab.core.errors import (
    ConfigError,
    DependencyMissing,
    PerceptionFailed,
    PhysLabError,
    PlanningFailed,
    SafetyViolation,
)
from physlab.core.events import Actor, EventLog
from physlab.core.logging import get_logger
from physlab.core.types import Mode, MotionResult, Pose, Pose2D
from physlab.experiments.analysis import fit_models, report
from physlab.experiments.protocol import ExperimentProtocol, validate_protocol
from physlab.experiments.registry import experiment_registry
from physlab.experiments.results import RunResult
from physlab.experiments.runner import ExperimentRunner
from physlab.fabrication.cad import build_part, list_parts
from physlab.fabrication.inventory import Inventory
from physlab.fabrication.mesh_check import check_printable, load_printer_profile
from physlab.fabrication.mock_print import MockPrintService
from physlab.fabrication.models import PartArtifact, PrintJob, PrintState
from physlab.hal.factory import create_camera, create_robot
from physlab.hal.interfaces import Arm, Camera, Robot
from physlab.mcp_server.envelope import for_log, jsonable
from physlab.mcp_server.jobs import JobManager, Progress
from physlab.mcp_server.models import (
    MoveArmJointsInput,
    MoveArmPoseInput,
    MoveBaseInput,
    PrintPartInput,
    RunChemInput,
    RunControllerInput,
)
from physlab.mcp_server.preview import render_stl_png
from physlab.mcp_server.scenes import list_scene_names, load_scene_file, scene_path
from physlab.perception.localize import Localizer
from physlab.robots.catalog import RobotCatalog
from physlab.safety.gate import gate
from physlab.sim.gl import close_renderers
from physlab.sim.world import SimWorld

_LOG = get_logger(__name__)
_SCENARIOS = {
    "focus_stage": ("focus_stage.py", "FocusStage"),
    "pendulum_release": ("pendulum_release.py", "PendulumRelease"),
    "line_follow": ("line_follow.py", "LineFollow"),
}


class _JumpClock:
    """Monotonic stand-in that finishes a mock print on the next status read."""

    def __init__(self) -> None:
        """Start at zero."""
        self.now_s = 0.0

    def __call__(self) -> float:
        """Advance one million seconds and return the new time.

        Returns:
            Simulated seconds.
        """
        self.now_s += 1.0e6
        return self.now_s


class LabSession:
    """Scene, inventory, designs, and the event log for one server process."""

    def __init__(self, config: PhysLabConfig, scene_name: str = "table_gantry") -> None:
        """Build the catalog and load ``scene_name``.

        Args:
            config: Effective physlab configuration. ``run_dir`` receives the
                event log, parts, and experiment runs.
            scene_name: Stem of a file in ``configs/scenes``.

        Raises:
            ConfigError: The scene file is missing or invalid.
            AssetMissing: The scene names a robot whose mesh is not on disk.
        """
        self._chem_lab = None
        self._chem_trials = []
        self.chem_bridge = None
        self.config = config
        self.catalog = RobotCatalog(config.configs_dir, config.assets_dir)
        self.event_log = EventLog(config.run_dir, mode=config.mode, seed=config.seed)
        self.jobs = JobManager()
        self.lock = threading.RLock()
        self.designs: dict[str, PartArtifact] = {}
        self.runs: dict[str, RunResult] = {}
        self._design_seq = 0
        self._robots: dict[str, Robot] = {}
        self._cameras: dict[str, Camera] = {}
        self.scene_name, self.world, self.printer, self.inventory = self._open_scene(scene_name)

    def record(self, action: str, params: dict[str, object], envelope: dict[str, object]) -> None:
        """Append one agent event. Image payloads and tokens are redacted.

        Args:
            action: Tool name.
            params: Tool inputs.
            envelope: Tool envelope.
        """
        safe_params = _as_dict(jsonable(_redact(params)))
        safe_result = _as_dict(jsonable(for_log(envelope)))
        with self.lock:
            self.event_log.record(
                Actor.AGENT,
                action,
                safe_params,
                safe_result,
                mode=self.config.mode,
            )

    def session_info(self) -> dict[str, object]:
        """Describe the session an agent is attached to.

        Returns:
            Mode, scene stem, robots, camera names, and the tool-call run id.
        """
        with self.lock:
            return {
                "mode": self.config.mode.value,
                "scene": self.scene_name,
                "robots": self._robot_rows(),
                "cameras": [camera.name for camera in self.world.scene.cameras],
                "run_id": self.event_log.run_id,
            }

    def list_scenes(self) -> dict[str, object]:
        """List scene stems.

        Returns:
            The loaded stem and every stem in ``configs/scenes``.
        """
        return {
            "loaded": self.scene_name,
            "scenes": list_scene_names(self.config.configs_dir),
        }

    def load(self, name: str) -> dict[str, object]:
        """Replace the world with ``name``. Designs on disk are kept.

        Spawned parts are dropped because their bodies belonged to the
        previous model. A new accelerated mock printer is attached.

        Args:
            name: Scene stem.

        Returns:
            The session description after the load.

        Raises:
            ConfigError: ``name`` is not a scene file.
        """
        with self.lock:
            close_renderers()
            self._robots.clear()
            self._cameras.clear()
            self.scene_name, self.world, self.printer, self.inventory = self._open_scene(name)
            self._chem_lab = None
            self._chem_trials = []
            self.chem_bridge = None
        _LOG.info("scene_loaded", scene=self.scene_name)
        return self.session_info()

    def reset(self, seed: int | None, approval_token: str | None) -> dict[str, object]:
        """Restore the initial state of the current scene.

        Args:
            seed: Seed stored on the world. The config seed is used when omitted.
            approval_token: Required in real mode.

        Returns:
            The seed that was applied and the simulation time, which is 0.

        Raises:
            ApprovalRequired: Real mode without a token.
            DependencyMissing: Real mode after the token check.
        """
        self._hardware("reset_scene", approval_token)
        chosen = self.config.seed if seed is None else seed
        with self.lock:
            self.world.reset(seed=chosen)
            self._chem_lab = None
            self._chem_trials = []
            if self.chem_bridge is not None:
                from physlab.chem.virtual_lab import VirtualChemLab

                self.chem_bridge.sync(VirtualChemLab())
                self.chem_bridge.frames.clear()
            time_s = self.world.time_s
        return {"seed": chosen, "time_s": time_s}

    def list_robots(self) -> dict[str, object]:
        """List robots placed in the current scene.

        Returns:
            One row per placement, with joint count and whether a gripper or
            base is declared.
        """
        with self.lock:
            return {"robots": self._robot_rows()}

    def robot_state(self, instance: str) -> dict[str, object]:
        """Read joints, the end effector, the gripper, and the base.

        Args:
            instance: Robot instance.

        Returns:
            State in the world frame. Joint positions use the same units as
            ``move_arm_joints``. Missing devices are null.

        Raises:
            ConfigError: ``instance`` is not in the scene.
        """
        with self.lock:
            robot = self._robot(instance)
            joints = None
            if robot.arm is not None:
                joints = robot.arm.get_joint_state().model_dump(mode="json")
            ee_pose = None
            if robot.arm is not None:
                ee_pose = robot.arm.get_ee_pose().model_dump(mode="json")
            opening = None if robot.gripper is None else robot.gripper.get_opening()
            grasping = None if robot.gripper is None else robot.gripper.is_grasping()
            base_pose = None
            if robot.base is not None:
                base_pose = robot.base.get_pose2d().model_dump(mode="json")
        return {
            "instance": instance,
            "robot_id": robot.spec.id,
            "joints": joints,
            "ee_pose": ee_pose,
            "gripper_opening": opening,
            "grasping": grasping,
            "base_pose": base_pose,
        }

    def move_joints(self, request: MoveArmJointsInput) -> dict[str, object]:
        """Move one arm in joint space.

        Args:
            request: Instance, targets, speed, and timeout.

        Returns:
            The motion result.

        Raises:
            ApprovalRequired: Real mode without a token.
            DependencyMissing: Real mode with a token.
            PlanningFailed: The motion did not reach the target.
            ConfigError: The instance has no arm.
        """
        self._hardware("move_arm_joints", request.approval_token)
        with self.lock:
            arm = self._arm(request.instance)
            result = arm.move_joints(
                request.q_rad,
                speed_scale=request.speed_scale,
                timeout_s=request.timeout_s,
            )
        return _require_motion(result)

    def move_pose(self, request: MoveArmPoseInput) -> dict[str, object]:
        """Move one arm to an end-effector pose.

        Args:
            request: Pose in ``request.frame``. Child-frame offsets used by
                grasp skills are not applied here.

        Returns:
            The motion result.

        Raises:
            ApprovalRequired: Real mode without a token.
            DependencyMissing: Real mode with a token.
            PlanningFailed: Inverse kinematics or the subsequent motion failed.
            ConfigError: The instance has no arm.
        """
        self._hardware("move_arm_pose", request.approval_token)
        target = Pose(
            position_m=request.position_m,
            quat_wxyz=request.quat_wxyz,
            frame=request.frame,
        )
        with self.lock:
            arm = self._arm(request.instance)
            result = arm.move_pose(
                target,
                speed_scale=request.speed_scale,
                timeout_s=request.timeout_s,
            )
        return _require_motion(result)

    def gripper(
        self,
        instance: str,
        action: str,
        force_scale: float,
        token: str | None,
    ) -> dict[str, object]:
        """Open or close the gripper.

        Args:
            instance: Robot instance.
            action: ``open`` or ``close``.
            force_scale: Close effort scale. The simulator ignores it.
            token: Approval token required in real mode.

        Returns:
            The motion result and the opening afterwards, from 0 closed to 1 open.

        Raises:
            ApprovalRequired: Real mode without a token.
            ConfigError: The robot has no gripper.
            PlanningFailed: The fingers did not finish the move.
        """
        self._hardware("gripper", token)
        with self.lock:
            robot = self._robot(instance)
            gripper = robot.gripper
            if gripper is None:
                raise ConfigError("Robot has no gripper", details={"instance": instance})
            result = gripper.open() if action == "open" else gripper.close(force_scale=force_scale)
            opening = gripper.get_opening()
        payload = _require_motion(result)
        payload["gripper_opening"] = opening
        return payload

    def move_base(self, request: MoveBaseInput) -> dict[str, object]:
        """Drive a mobile base to a planar pose in the world frame.

        Args:
            request: ``x_m``, ``y_m``, and ``yaw_rad``.

        Returns:
            The motion result.

        Raises:
            ApprovalRequired: Real mode without a token.
            ConfigError: The robot has no base.
            PlanningFailed: The base did not arrive.
        """
        self._hardware("move_base", request.approval_token)
        target = Pose2D(x_m=request.x_m, y_m=request.y_m, yaw_rad=request.yaw_rad, frame="world")
        with self.lock:
            robot = self._robot(request.instance)
            if robot.base is None:
                raise ConfigError(
                    "Robot has no mobile base",
                    details={"instance": request.instance},
                )
            result = robot.base.move_to(target)
        return _require_motion(result)

    def stop_all(self) -> dict[str, object]:
        """Hold every arm and stop every base in the scene.

        Stopping does not require an approval token.

        Returns:
            The device names that were commanded.
        """
        stopped: list[str] = []
        with self.lock:
            for placement in self.world.scene.robots:
                robot = self._robot(placement.instance)
                if robot.arm is not None:
                    robot.arm.stop()
                    stopped.append(f"{placement.instance}.arm")
                if robot.base is not None:
                    robot.base.stop()
                    stopped.append(f"{placement.instance}.base")
        return {"stopped": stopped}

    def capture(self, camera_name: str) -> tuple[dict[str, object], bytes]:
        """Grab one RGB frame and encode it as PNG.

        Args:
            camera_name: Camera in the scene.

        Returns:
            Metadata (no pixel array) and PNG bytes. Depth is summarized by
            its minimum and maximum in meters.

        Raises:
            ConfigError: The camera is not in the scene.
            PerceptionFailed: The PNG encoder failed.
        """
        with self.lock:
            frame = self._camera(camera_name).capture()
        image = np.ascontiguousarray(frame.rgb)
        encoded, buffer = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not encoded:
            raise PerceptionFailed("PNG encode failed", details={"camera": camera_name})
        depth = None
        if frame.depth_m is not None:
            depth = {
                "min_m": float(np.min(frame.depth_m)),
                "max_m": float(np.max(frame.depth_m)),
            }
        pose = None if frame.pose_in_world is None else frame.pose_in_world.model_dump(mode="json")
        meta: dict[str, object] = {
            "camera": camera_name,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "timestamp_s": frame.timestamp_s,
            "intrinsics": frame.intrinsics.model_dump(mode="json"),
            "pose_in_world": pose,
            "depth_m": depth,
        }
        return meta, buffer.tobytes()

    def locate_marker(self, marker_id: int, marker_size_m: float) -> dict[str, object]:
        """Fuse an ArUco marker into the world frame.

        Args:
            marker_id: Marker id.
            marker_size_m: Outer marker side, meters.

        Returns:
            ``found`` false when no camera sees the marker. A pose is in the
            world frame, quaternion ``(w, x, y, z)``.

        Raises:
            PerceptionFailed: The scene has no cameras.
        """
        with self.lock:
            if not self.world.scene.cameras:
                raise PerceptionFailed("Scene has no cameras")
            cameras = [self._camera(spec.name) for spec in self.world.scene.cameras]
            located = Localizer(cameras, marker_size_m).locate(marker_id)
        if located is None:
            return {"found": False, "marker_id": marker_id, "pose_world": None}
        return {
            "found": True,
            "marker_id": marker_id,
            "pose_world": located.pose_world.model_dump(mode="json"),
            "n_views": located.n_views,
            "residual_m": located.residual_m,
        }

    def list_parts(self) -> dict[str, object]:
        """List library parts and their parameter schemas.

        Returns:
            Names and JSON schemas. Schema lengths are millimeters.
        """
        return {"parts": [item.model_dump(mode="json") for item in list_parts()]}

    def design(self, part_name: str, params: dict[str, object]) -> tuple[dict[str, object], bytes]:
        """Build a part and render a preview PNG from its STL.

        Args:
            part_name: Library name.
            params: Parameter overrides. Units follow the part schema.

        Returns:
            Artifact summary, including a base64 preview, and the PNG bytes.

        Raises:
            ConfigError: The name or the parameters are invalid.
        """
        directory = self.event_log.path.parent / "parts"
        artifact = build_part(part_name, params, directory)
        png = render_stl_png(artifact.stl_path)
        with self.lock:
            self._design_seq += 1
            design_id = f"design-{self._design_seq}"
            self.designs[design_id] = artifact
        return _artifact_summary(design_id, artifact, png), png

    def check_part(self, design_id: str) -> dict[str, object]:
        """Check a design against the printer profile.

        Args:
            design_id: Id from :meth:`design`.

        Returns:
            Printability report. Lengths in the profile are millimeters.
            ``bbox`` on the design stays in meters.

        Raises:
            ConfigError: ``design_id`` is unknown.
        """
        artifact = self._design(design_id)
        checked = check_printable(artifact.stl_path, load_printer_profile())
        return checked.model_dump(mode="json")

    def print_part(self, request: PrintPartInput, progress: Progress) -> dict[str, object]:
        """Mock-print a design into the pickup zone.

        The sim clock jumps, so the job reaches ``DONE`` on the first status
        read. The part is spawned with an ArUco id when the world is live.

        Args:
            request: Design, material, quality, and copies.
            progress: Job progress callback.

        Returns:
            Print status and the inventory row.

        Raises:
            ApprovalRequired: Real mode without a token.
            DependencyMissing: Real mode with a token.
            ConfigError: The design is unknown or the print did not finish.
        """
        self._hardware("print_part", request.approval_token)
        artifact = self._design(request.design_id)
        with self.lock:
            job_id = self.printer.submit(
                PrintJob(
                    part=artifact,
                    material=request.material,
                    quality=request.quality,
                    copies=request.copies,
                )
            )
            status = self.printer.status(job_id)
            progress(status.progress)
            item = self.printer.inventory_entry(job_id)
        if status.state is not PrintState.DONE or item is None:
            raise ConfigError(
                "Mock print did not finish",
                details={"state": status.state.value, "message": status.message},
            )
        return {
            "print_job_id": job_id,
            "design_id": request.design_id,
            "status": status.model_dump(mode="json"),
            "inventory": item.model_dump(mode="json"),
        }

    def list_inventory(self) -> dict[str, object]:
        """List printed copies currently in the session.

        Returns:
            Inventory rows. Poses are in the world frame, meters.
        """
        with self.lock:
            rows = [item.model_dump(mode="json") for item in self.inventory.items()]
        return {"items": rows}

    def validate_assembly(self, plan_json: dict[str, object]) -> dict[str, object]:
        """Check an assembly plan against the inventory.

        Args:
            plan_json: Plan object. Place targets are meters in the world frame.

        Returns:
            ``valid`` and the issue strings. An empty issue list can still
            fail at execution when the marker is not seen.

        Raises:
            ConfigError: The JSON is not an assembly plan.
        """
        plan = _parse(AssemblyPlan, plan_json, "assembly plan")
        with self.lock:
            issues = validate_plan(plan, self.inventory, self.catalog)
        return {"valid": not issues, "issues": issues}

    def run_assembly(
        self,
        plan_json: dict[str, object],
        approval_token: str | None,
        progress: Progress,
    ) -> dict[str, object]:
        """Execute an assembly plan on the first robot that has a gripper.

        Args:
            plan_json: Plan object.
            approval_token: Required in real mode.
            progress: Job progress callback.

        Returns:
            Assembly report. ``failed_step`` is null when every step succeeded.

        Raises:
            ApprovalRequired: Real mode without a token.
            DependencyMissing: Real mode with a token.
            ConfigError: The plan is invalid, a component was not designed, or
                the scene has no gripper.
        """
        self._hardware("run_assembly_plan", approval_token)
        plan = _parse(AssemblyPlan, plan_json, "assembly plan")
        with self.lock:
            issues = validate_plan(plan, self.inventory, self.catalog)
            if issues:
                raise ConfigError("Assembly plan is invalid", details={"issues": issues})
            robot = self._gripper_robot()
            localizer = Localizer(
                [self._camera("overhead_left"), self._camera("overhead_right")],
                0.03,
            )
            artifacts = self._artifacts(plan)
            progress(0.1)
            finished = AssemblyExecutor(
                robot,
                localizer,
                self.inventory,
                self.event_log,
                self.world,
                self.catalog,
                artifacts,
            ).run(plan)
            progress(1.0)
        return finished.model_dump(mode="json")

    def device_manifest(self, scenario: str) -> dict[str, object]:
        """Return the sensor and actuator channels for a scenario.

        Args:
            scenario: ``focus_stage``, ``pendulum_release``, or ``line_follow``.

        Returns:
            Manifest. Each channel names its unit.

        Raises:
            ConfigError: The scenario name is unknown.
        """
        plant = self._scenario(scenario)
        return plant.manifest.model_dump(mode="json")

    def check_code(self, source: str) -> dict[str, object]:
        """Run the static controller allowlist.

        Args:
            source: Python source.

        Returns:
            ``ok`` and the violations. A failure here is not an exception.
        """
        return check_source(source).model_dump(mode="json")

    def run_controller(self, request: RunControllerInput, progress: Progress) -> dict[str, object]:
        """Run a controller in the process sandbox.

        Args:
            request: Source, scenario, duration, params, and seed.
            progress: Job progress callback.

        Returns:
            Run report plus ``plot_images`` as PNG base64. ``summary_text`` is
            capped at 2000 characters. A failed static check is a finished
            report with ``ok`` false and ``phase_failed`` ``CHECK``.

        Raises:
            ApprovalRequired: Real mode without a token.
            DependencyMissing: Real mode with a token.
            ConfigError: The scenario name is unknown.
        """
        self._hardware("run_controller", request.approval_token)
        plant = self._scenario(request.scenario)
        directory = self.event_log.path.parent / "controllers"
        progress(0.05)
        finished = run_controller(
            request.source,
            plant.manifest,
            plant,
            request.duration_s,
            request.params,
            request.seed,
            output_dir=directory,
        )
        progress(1.0)
        payload = finished.model_dump(mode="json")
        images: list[dict[str, object]] = []
        for plot in finished.plots:
            images.append(
                {
                    "path": str(plot),
                    "mime_type": "image/png",
                    "base64": base64.standard_b64encode(plot.read_bytes()).decode("ascii"),
                }
            )
        payload["plot_images"] = images
        return payload

    def validate_experiment(self, protocol_json: dict[str, object]) -> dict[str, object]:
        """Check an experiment protocol against the skill registry.

        Args:
            protocol_json: Protocol object. Variable units are named by the author.

        Returns:
            ``valid`` and the issue strings.

        Raises:
            ConfigError: The JSON is not a protocol.
        """
        protocol = _parse(ExperimentProtocol, protocol_json, "experiment protocol")
        issues = validate_protocol(protocol, experiment_registry().names())
        return {"valid": not issues, "issues": issues}

    def run_experiment(
        self,
        protocol_json: dict[str, object],
        approval_token: str | None,
        progress: Progress,
    ) -> dict[str, object]:
        """Run an experiment protocol and keep the run for fitting.

        Args:
            protocol_json: Protocol object.
            approval_token: Required in real mode.
            progress: Job progress callback.

        Returns:
            Run id, trial count, and how many trials succeeded.

        Raises:
            ApprovalRequired: Real mode without a token.
            DependencyMissing: Real mode with a token.
            ConfigError: The protocol is invalid or names an unknown skill.
        """
        self._hardware("run_experiment", approval_token)
        protocol = _parse(ExperimentProtocol, protocol_json, "experiment protocol")
        log = EventLog(self.config.run_dir, mode=self.config.mode, seed=protocol.seed)
        runner = ExperimentRunner(log, world=self.world)
        progress(0.1)
        with self.lock:
            result = runner.run(protocol)
            self.runs[result.run_id] = result
        progress(1.0)
        return _run_summary(result)

    def check_chem(self, protocol_json: dict[str, object], token: str | None) -> dict[str, object]:
        """Check volumes and the reagent allowlist.

        Args:
            protocol_json: Chemistry protocol. Concentrations are molar and
                volumes are microliters.
            token: Forwarded to the allowlist check. Real mode without a token
                is a denial, not an exception, so an agent can read the reason.

        Returns:
            Structural issues and the safety decision.

        Raises:
            ConfigError: The JSON is not a chemistry protocol.
        """
        protocol = _parse(ChemProtocol, protocol_json, "chemistry protocol")
        with self.lock:
            if self._chem_lab is not None and not protocol.reagents:
                protocol = self._chem_lab.combined(protocol)
        issues = validate_chem_protocol(protocol)
        decision = ChemSafetyPolicy.load().check_protocol(
            protocol,
            mode=self.config.mode,
            approval_token=token,
        )
        return {
            "valid": not issues and decision.allowed,
            "issues": issues,
            "safety": decision.model_dump(mode="json"),
        }

    def run_chem(self, request: RunChemInput, progress: Progress) -> dict[str, object]:
        """Execute a chemistry protocol on the virtual lab and store the run.

        Args:
            request: Protocol, question, hypothesis, seed, and analyte.
            progress: Job progress callback.

        Returns:
            Run summary. Readings are one trial each so ``fit_models`` can
            use the table.

        Raises:
            ApprovalRequired: Real mode without a token.
            SafetyViolation: The allowlist denied the protocol.
            ConfigError: The protocol does not fit the labware.
        """
        with self.lock:
            return self._run_chem_locked(request, progress)

    def _run_chem_locked(self, request: RunChemInput, progress: Progress) -> dict[str, object]:
        self._hardware("run_chem_protocol", request.approval_token)
        protocol = _parse(ChemProtocol, request.protocol, "chemistry protocol")
        decision = ChemSafetyPolicy.load().check_protocol(
            protocol,
            mode=self.config.mode,
            approval_token=request.approval_token,
        )
        if not decision.allowed:
            raise SafetyViolation(
                "Chemistry protocol was denied",
                details={"violations": decision.violations},
            )
        log = EventLog(self.config.run_dir, mode=self.config.mode, seed=request.seed)
        progress(0.2)
        from physlab.chem.gantry_bridge import GantryBridge
        from physlab.chem.protocol import TransferStep
        from physlab.chem.virtual_lab import VirtualChemLab

        if self._chem_lab is None:
            self._chem_lab = VirtualChemLab(seed=request.seed, analyte=request.analyte)
        before = copy.deepcopy(self._chem_lab)
        result = ExperimentRunner(log).run_chem(
            protocol,
            lab=self._chem_lab,
            reset=bool(protocol.reagents),
            question=request.question,
            hypothesis=request.hypothesis,
            seed=request.seed,
            analyte=request.analyte,
        )
        with self.lock:
            if protocol.reagents:
                self._chem_trials = []
            self._chem_trials.extend(result.trials)
            result.trials[:] = self._chem_trials
            from physlab.experiments.storage import write_trials

            write_trials(result.directory, result.trials)
            if self.world.scene.chem_deck:
                if self.chem_bridge is None:
                    self.chem_bridge = GantryBridge(self.world)
                if protocol.reagents:
                    before._load(protocol)
                for step in protocol.steps:
                    previous = copy.deepcopy(before)
                    before._apply(protocol, step)
                    if isinstance(step, TransferStep):
                        self.chem_bridge.animate(step, previous, before)
                self.chem_bridge.sync(self._chem_lab)
            self.runs[result.run_id] = result
        progress(1.0)
        return _run_summary(result)

    def fit(self, run_id: str, x: str, y: str) -> dict[str, object]:
        """Fit the six built-in curves to one run and sort them by AIC.

        Args:
            run_id: Run id from an experiment or chemistry job.
            x: Independent column.
            y: Dependent column.

        Returns:
            Fits, lowest AIC first. Parameters are dimensionless coefficients
            of the named formula.

        Raises:
            ConfigError: The run or a column is missing.
        """
        finished = self._require_run(run_id)
        fits = fit_models(_frame(finished), x, y)
        return {"run_id": run_id, "fits": [item.model_dump(mode="json") for item in fits]}

    def write_report(self, run_id: str, x: str | None, y: str | None) -> dict[str, object]:
        """Fit ``x`` against ``y`` and write the Markdown report.

        Args:
            run_id: Run id.
            x: Independent column. Inferred when a successful trial has one value.
            y: Dependent column. Inferred when that trial has one measurement.

        Returns:
            Report path and the Markdown text.

        Raises:
            ConfigError: The run is missing or the columns cannot be inferred.
        """
        finished = self._require_run(run_id)
        x_name, y_name = _axes(finished, x, y)
        fits = fit_models(_frame(finished), x_name, y_name)
        path = report(finished, fits, x_name, y_name)
        return {
            "run_id": run_id,
            "path": str(path),
            "x": x_name,
            "y": y_name,
            "markdown": path.read_text(encoding="utf-8"),
        }

    def report_text(self, run_id: str) -> str:
        """Read a report written by :meth:`write_report`.

        Args:
            run_id: Single path segment. The file is ``run_dir/run_id/report.md``.

        Returns:
            Markdown text.

        Raises:
            ConfigError: The id is unsafe or the report does not exist.
            SafetyViolation: The resolved path leaves ``run_dir``.
        """
        path = self._report_path(run_id)
        if not path.is_file():
            raise ConfigError("No report has been written for this run", details={"run_id": run_id})
        return path.read_text(encoding="utf-8")

    def agent_guide(self) -> str:
        """Return the agent guide shipped next to the configs directory.

        Returns:
            Markdown text.

        Raises:
            ConfigError: ``AGENT_GUIDE.md`` is not in the project root.
        """
        path = self.config.configs_dir.parent / "AGENT_GUIDE.md"
        if not path.is_file():
            raise ConfigError("AGENT_GUIDE.md was not found", details={"path": str(path)})
        return path.read_text(encoding="utf-8")

    def catalog_document(self) -> dict[str, object]:
        """Return the robot catalog as a JSON document.

        Returns:
            Every catalog entry. ``available`` is false when the mesh is missing.
        """
        rows = []
        for spec in self.catalog.list():
            rows.append(
                {
                    "id": spec.id,
                    "display_name": spec.display_name,
                    "kind": spec.kind.value,
                    "available": spec.available,
                    "dof": len(spec.sim.arm_joints),
                    "license": spec.license,
                }
            )
        return {"robots": rows}

    def _hardware(self, action: str, approval_token: str | None) -> None:
        gate(action, mode=self.config.mode, approval_token=approval_token)
        if self.config.mode is Mode.REAL:
            raise DependencyMissing(
                "Real hardware backends are not installed",
                details={"action": action},
            )

    def _robot_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for placement in self.world.scene.robots:
            spec = self.catalog.get(placement.robot_id)
            rows.append(
                {
                    "instance": placement.instance,
                    "robot_id": placement.robot_id,
                    "kind": spec.kind.value,
                    "dof": len(spec.sim.arm_joints),
                    "has_gripper": spec.sim.gripper is not None,
                    "has_base": bool(spec.sim.base_joints),
                }
            )
        return rows

    def _robot(self, instance: str) -> Robot:
        cached = self._robots.get(instance)
        if cached is None:
            cached = create_robot(self.world, instance)
            self._robots[instance] = cached
        return cached

    def _arm(self, instance: str) -> Arm:
        arm = self._robot(instance).arm
        if arm is None:
            raise ConfigError("Robot has no arm", details={"instance": instance})
        return arm

    def _camera(self, name: str) -> Camera:
        cached = self._cameras.get(name)
        if cached is None:
            cached = create_camera(self.world, name)
            self._cameras[name] = cached
        return cached

    def _design(self, design_id: str) -> PartArtifact:
        with self.lock:
            artifact = self.designs.get(design_id)
        if artifact is None:
            raise ConfigError("Unknown design", details={"design_id": design_id})
        return artifact

    def _gripper_robot(self) -> Robot:
        for placement in self.world.scene.robots:
            robot = self._robot(placement.instance)
            if robot.arm is not None and robot.gripper is not None:
                return robot
        raise ConfigError("Scene has no robot with an arm and a gripper")

    def _artifacts(self, plan: AssemblyPlan) -> dict[str, PartArtifact]:
        found: dict[str, PartArtifact] = {}
        for component in plan.components:
            match = next(
                (
                    artifact
                    for artifact in self.designs.values()
                    if artifact.name == component.part_name
                ),
                None,
            )
            if match is None:
                raise ConfigError(
                    "No design for assembly component",
                    details={"component": component.id, "part_name": component.part_name},
                )
            found[component.id] = match
        return found

    def _scenario(self, name: str) -> Scenario:
        entry = _SCENARIOS.get(name)
        if entry is None:
            raise ConfigError(
                "Unknown scenario",
                details={"scenario": name, "known": sorted(_SCENARIOS)},
            )
        filename, class_name = entry
        path = self.config.configs_dir.parent / "examples" / "scenarios" / filename
        if not path.is_file():
            raise ConfigError("Scenario file is missing", details={"path": str(path)})
        spec = importlib.util.spec_from_file_location(f"physlab_scenario_{name}", path)
        if spec is None or spec.loader is None:
            raise ConfigError("Scenario module could not be loaded", details={"path": str(path)})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        factory = getattr(module, class_name, None)
        if not callable(factory):
            raise ConfigError("Scenario class is missing", details={"class": class_name})
        return cast(Scenario, factory())

    def _require_run(self, run_id: str) -> RunResult:
        with self.lock:
            finished = self.runs.get(run_id)
        if finished is None:
            raise ConfigError("Unknown run", details={"run_id": run_id})
        return finished

    def _open_scene(self, name: str) -> tuple[str, SimWorld, MockPrintService, Inventory]:
        """Compile ``name`` and attach an accelerated mock printer.

        Args:
            name: Scene stem.

        Returns:
            Scene name, world, printer, and the printer's inventory.
        """
        path = scene_path(self.config.configs_dir, name)
        parsed_name, spec = load_scene_file(path)
        world = SimWorld(spec, self.catalog, self.config)
        printer = MockPrintService(world, mm3_per_s=1.0e9, clock=_JumpClock())
        return parsed_name, world, printer, printer.inventory

    def _report_path(self, run_id: str) -> Path:
        if not run_id or run_id != Path(run_id).name or ".." in run_id:
            raise ConfigError("run_id must be a single path segment", details={"run_id": run_id})
        root = self.config.run_dir.resolve()
        path = (self.config.run_dir / run_id / "report.md").resolve()
        if path.parent.parent != root:
            raise SafetyViolation(
                "Report path escaped the run directory",
                details={"path": str(path)},
            )
        return path


def _require_motion(result: MotionResult) -> dict[str, object]:
    payload = _as_dict(jsonable(result.model_dump(mode="json")))
    if not result.success:
        raise PlanningFailed(result.message or "motion failed", details=payload)
    return payload


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _parse(model: type[_ModelT], data: dict[str, object], label: str) -> _ModelT:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid {label}", details={"errors": str(exc)}) from exc


def _artifact_summary(design_id: str, artifact: PartArtifact, png: bytes) -> dict[str, object]:
    frames = {name: pose.model_dump(mode="json") for name, pose in artifact.frames.items()}
    return {
        "design_id": design_id,
        "name": artifact.name,
        "params": artifact.params,
        "stl_path": str(artifact.stl_path),
        "step_path": str(artifact.step_path),
        "bbox_m": list(artifact.bbox_m),
        "frames": frames,
        "preview_png_base64": base64.standard_b64encode(png).decode("ascii"),
    }


def _run_summary(result: RunResult) -> dict[str, object]:
    succeeded = sum(1 for trial in result.trials if trial.success)
    return {
        "run_id": result.run_id,
        "protocol_id": result.protocol_id,
        "directory": str(result.directory),
        "trials": len(result.trials),
        "succeeded": succeeded,
    }


def _frame(result: RunResult) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for trial in result.trials:
        if not trial.success:
            continue
        row: dict[str, float] = {}
        row.update(trial.values)
        row.update(trial.measurements)
        rows.append(row)
    if not rows:
        raise ConfigError("Run has no successful trials", details={"run_id": result.run_id})
    return pd.DataFrame(rows)


def _axes(result: RunResult, x: str | None, y: str | None) -> tuple[str, str]:
    if x is not None and y is not None:
        return x, y
    trial = next((item for item in result.trials if item.success), None)
    if trial is None:
        raise ConfigError("Run has no successful trial", details={"run_id": result.run_id})
    if x is None:
        if len(trial.values) != 1:
            raise ConfigError("Pass x; the trial has more than one value column")
        x = next(iter(trial.values))
    if y is None:
        if len(trial.measurements) != 1:
            raise ConfigError("Pass y; the trial has more than one measurement")
        y = next(iter(trial.measurements))
    return x, y


def _as_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return cast(dict[str, object], value)
    raise PhysLabError("Expected a JSON object", details={"type": type(value).__name__})


def _redact(params: dict[str, object]) -> dict[str, object]:
    if not params.get("approval_token"):
        return params
    copied = dict(params)
    copied["approval_token"] = "***"
    return copied
