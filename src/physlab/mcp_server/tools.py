"""Register every physlab MCP tool and resource.

Tools take one pydantic ``request`` object. They return ``{ok, data, error}``.
``capture_image`` and ``design_part`` also attach a PNG image block.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import cast

from mcp.server.mcpserver import Image, MCPServer
from mcp.types import CallToolResult, ContentBlock, TextContent
from pydantic import BaseModel

from physlab.experiments.registry import experiment_registry
from physlab.mcp_server.envelope import failure, success
from physlab.mcp_server.jobs import Progress
from physlab.mcp_server.models import (
    AssemblyPlanInput,
    CaptureImageInput,
    CheckCodeInput,
    CheckPartInput,
    DesignPartInput,
    FitModelsInput,
    GetJobInput,
    GetReportInput,
    GripperInput,
    LoadSceneInput,
    LocateMarkerInput,
    MoveArmJointsInput,
    MoveArmPoseInput,
    MoveBaseInput,
    PrintPartInput,
    ProtocolInput,
    ResetSceneInput,
    RobotInstanceInput,
    RunChemInput,
    RunControllerInput,
    RunExperimentInput,
    ScenarioInput,
    StopAllInput,
)
from physlab.mcp_server.session import LabSession
from physlab.safety.gate import gate

_ASSEMBLY_OPS = ("pick", "place", "fasten", "align", "inspect")
_MOTION = ("safe_transit", "approach", "retreat", "navigate_and_reach")


def register_tools(server: MCPServer, session: LabSession) -> None:
    """Attach every tool. A missing name here is an incomplete server.

    Args:
        server: MCP server that will publish the tools.
        session: Session the tools close over.
    """

    @server.tool()
    def get_session_info() -> dict[str, object]:
        """Return mode, scene, robots, and cameras for this process.

        Units are unused. Poses from other tools use the world frame.
        Fails only when the session could not be built.
        Example: get_session_info().

        Returns:
            Envelope. ``data`` contains mode, scene, robots, cameras, run_id.
        """
        return _ok(session, "get_session_info", None, session.session_info)

    @server.tool()
    def reset_scene(request: ResetSceneInput) -> dict[str, object]:
        """Restore the loaded scene to its initial state.

        ``seed`` seeds the simulator. Time goes back to 0 seconds. The world
        frame is unchanged. Real mode requires ``approval_token`` and then
        refuses because real drivers are not installed.
        Example: reset_scene(request={"seed": 0}).

        Args:
            request: Optional seed and approval token.

        Returns:
            Envelope with the seed that was applied.
        """
        return _ok(
            session,
            "reset_scene",
            request,
            lambda: session.reset(request.seed, request.approval_token),
        )

    @server.tool()
    def list_scenes() -> dict[str, object]:
        """List scene file stems under configs/scenes.

        Names are not filesystem paths. Loading one rebuilds the world frame.
        Fails only if the scene directory cannot be read.
        Example: list_scenes().

        Returns:
            Envelope with ``loaded`` and ``scenes``.
        """
        return _ok(session, "list_scenes", None, session.list_scenes)

    @server.tool()
    def load_scene(request: LoadSceneInput) -> dict[str, object]:
        """Replace the world with a named scene file.

        Robot bases are meters in the world frame. Spawned parts are dropped.
        Designs already built stay on disk. Fails when the stem is unknown or
        a robot mesh is missing.
        Example: load_scene(request={"name": "table_gantry"}).

        Args:
            request: Scene stem.

        Returns:
            Envelope with the new session description.
        """
        return _ok(session, "load_scene", request, lambda: session.load(request.name))

    @server.tool()
    def list_robots() -> dict[str, object]:
        """List robots placed in the current scene.

        ``dof`` counts arm joints. Gantry joints are meters; revolute joints
        are radians. The world frame is the scene frame. Fails if the scene
        did not compile.
        Example: list_robots().

        Returns:
            Envelope with one row per placed robot.
        """
        return _ok(session, "list_robots", None, session.list_robots)

    @server.tool()
    def get_robot_state(request: RobotInstanceInput) -> dict[str, object]:
        """Read joints, end effector, gripper, and base for one instance.

        End-effector and base poses are in the world frame, meters and radians.
        Joint positions use the arm's native units. Fails when the instance is
        not in the scene.
        Example: get_robot_state(request={"instance": "gantry"}).

        Args:
            request: Robot instance name.

        Returns:
            Envelope with joints, ee_pose, gripper_opening, and base_pose.
        """
        return _ok(
            session,
            "get_robot_state",
            request,
            lambda: session.robot_state(request.instance),
        )

    @server.tool()
    def move_arm_joints(request: MoveArmJointsInput) -> dict[str, object]:
        """Move one arm to a joint-space target.

        Revolute targets are radians. Prismatic gantry targets are meters.
        On the procedural gantry, an X-only target settles; Y or Z can
        overshoot and time out. The end effector ends in the world frame.
        Real mode requires
        ``approval_token``. Fails on joint limits, collision, or timeout.
        Example: move_arm_joints(request={"instance": "gantry", "q_rad": [0.05, 0.0, 0.02]}).

        Args:
            request: Instance, targets, speed scale, timeout in seconds, token.

        Returns:
            Envelope whose data is the motion result.
        """
        return _ok(session, "move_arm_joints", request, lambda: session.move_joints(request))

    @server.tool()
    def move_arm_pose(request: MoveArmPoseInput) -> dict[str, object]:
        """Move one arm so the end effector matches a pose.

        ``position_m`` is meters in ``frame`` (usually ``world``).
        ``quat_wxyz`` is ``(w, x, y, z)``. Real mode requires a token.
        Fails when inverse kinematics or the joint motion fails.
        Example: move_arm_pose(request={"instance": "arm", "position_m": [0.4, 0.0, 0.9]}).

        Args:
            request: Instance, pose, speed, timeout, and token.

        Returns:
            Envelope whose data is the motion result.
        """
        return _ok(session, "move_arm_pose", request, lambda: session.move_pose(request))

    @server.tool()
    def gripper(request: GripperInput) -> dict[str, object]:
        """Open or close the parallel gripper.

        ``force_scale`` is dimensionless. Opening is 0 when closed and 1 when
        open, in the gripper frame. Real mode requires a token. Fails when the
        robot has no gripper or the fingers do not finish.
        Example: gripper(request={"instance": "arm", "action": "close"}).

        Args:
            request: Instance, ``open`` or ``close``, force scale, and token.

        Returns:
            Envelope with the motion result and ``gripper_opening``.
        """
        return _ok(
            session,
            "gripper",
            request,
            lambda: session.gripper(
                request.instance,
                request.action,
                request.force_scale,
                request.approval_token,
            ),
        )

    @server.tool()
    def move_base(request: MoveBaseInput) -> dict[str, object]:
        """Drive a mobile base to a planar pose in the world frame.

        ``x_m`` and ``y_m`` are meters. ``yaw_rad`` is heading about world Z.
        Real mode requires a token. Fails when the robot has no base or the
        motion times out.
        Example: move_base(request={"instance": "base", "x_m": 0.2, "y_m": 0.0, "yaw_rad": 0.0}).

        Args:
            request: Instance, planar target, and token.

        Returns:
            Envelope whose data is the motion result.
        """
        return _ok(session, "move_base", request, lambda: session.move_base(request))

    @server.tool()
    def stop_all(request: StopAllInput | None = None) -> dict[str, object]:
        """Hold every arm and zero every base command.

        This is the safe stop. It does not require an approval token in real
        mode. Joints stay in the world frame. Fails only if a device raises
        while halting.
        Example: stop_all(request={}).

        Args:
            request: Unused token field, accepted so the schema stays a model.

        Returns:
            Envelope listing the devices that were stopped.
        """
        return _ok(session, "stop_all", request, session.stop_all)

    @server.tool()
    def capture_image(request: CaptureImageInput) -> CallToolResult:
        """Capture one camera as a PNG plus metadata.

        The image is RGB. ``pose_in_world`` is the camera in the world frame,
        meters and ``(w, x, y, z)``. Depth is summarized in meters. Fails when
        the camera name is unknown or the encoder fails.
        Example: capture_image(request={"camera": "overhead_left"}).

        Args:
            request: Camera name.

        Returns:
            Text envelope plus one ``image/png`` block when the grab succeeds.
        """
        return _png(session, "capture_image", request, lambda: session.capture(request.camera))

    @server.tool()
    def locate_marker(request: LocateMarkerInput) -> dict[str, object]:
        """Fuse an ArUco marker into the world frame.

        ``marker_size_m`` is the outer side in meters, border included.
        ``pose_world`` uses ``(w, x, y, z)``. A marker that no camera sees is
        ``found: false``, not an exception. Fails when the scene has no cameras.
        Example: locate_marker(request={"marker_id": 31, "marker_size_m": 0.03}).

        Args:
            request: Marker id and size.

        Returns:
            Envelope with ``found``, ``pose_world``, ``n_views``, and residual meters.
        """
        return _ok(
            session,
            "locate_marker",
            request,
            lambda: session.locate_marker(request.marker_id, request.marker_size_m),
        )

    @server.tool()
    def list_parts() -> dict[str, object]:
        """List parametric parts and the JSON schema of each parameter set.

        Schema lengths are millimeters. Built artifacts report meters.
        Fails only if the library failed to import.
        Example: list_parts().

        Returns:
            Envelope with ``parts``.
        """
        return _ok(session, "list_parts", None, session.list_parts)

    @server.tool()
    def design_part(request: DesignPartInput) -> CallToolResult:
        """Build a part and attach a matplotlib preview of the STL.

        Parameters are millimeters. ``bbox_m`` and ``frames`` are meters in
        the part body frame (``frame`` is ``body``). Fails on an unknown name
        or parameters that miss the schema.
        Example: design_part(request={"part_name": "lens_holder", "params": {}}).

        Args:
            request: Library name and parameter overrides.

        Returns:
            Text envelope plus one ``image/png`` preview.
        """
        return _png(
            session,
            "design_part",
            request,
            lambda: session.design(request.part_name, request.params),
        )

    @server.tool()
    def check_part(request: CheckPartInput) -> dict[str, object]:
        """Check a designed STL against the printer profile.

        The profile uses millimeters. The mesh must be watertight and fit the
        bed. Fails when ``design_id`` is unknown.
        Example: check_part(request={"design_id": "design-1"}).

        Args:
            request: Design id from ``design_part``.

        Returns:
            Envelope with the printability report.
        """
        return _ok(session, "check_part", request, lambda: session.check_part(request.design_id))

    @server.tool()
    def print_part(request: PrintPartInput) -> dict[str, object]:
        """Queue a mock print and return a job id.

        Simulation finishes on the next status read and spawns the mesh in the
        world frame at the pickup pose, meters. Real mode requires a token and
        does not print. Poll ``get_job``. Fails when the design is unknown.
        Example: print_part(request={"design_id": "design-1"}).

        Args:
            request: Design id, material, quality, copies, and token.

        Returns:
            Envelope whose data is ``{"job_id": ...}``.
        """

        def work(progress: Progress) -> object:
            return session.print_part(request, progress)

        return _guarded_job(session, "print_part", request, request.approval_token, work)

    @server.tool()
    def list_inventory() -> dict[str, object]:
        """List printed copies in this session.

        Poses are meters in the world frame. ``aruco_id`` is the marker to pass
        to an assembly plan. Fails only if the inventory cannot be read.
        Example: list_inventory().

        Returns:
            Envelope with ``items``.
        """
        return _ok(session, "list_inventory", None, session.list_inventory)

    @server.tool()
    def validate_assembly_plan(request: AssemblyPlanInput) -> dict[str, object]:
        """Check an assembly plan without moving the robot.

        Place targets are meters in the world frame, quaternion ``(w, x, y, z)``.
        A component must already be in the inventory with the same part name and
        ArUco id. Fails when the JSON is not a plan.
        Example: validate_assembly_plan(request={"plan": {"id": "holder"}}).

        Args:
            request: Plan object. The token is ignored.

        Returns:
            Envelope with ``valid`` and ``issues``.
        """
        return _ok(
            session,
            "validate_assembly_plan",
            request,
            lambda: session.validate_assembly(request.plan),
        )

    @server.tool()
    def run_assembly_plan(request: AssemblyPlanInput) -> dict[str, object]:
        """Run an assembly plan on a background job.

        Targets are meters in the world frame. The robot needs an arm, a
        gripper, and the overhead stereo pair. Real mode requires a token.
        Poll ``get_job``. Fails fast when the plan or the scene cannot run.
        Example: run_assembly_plan(request={"plan": {"id": "holder"}}).

        Args:
            request: Plan object and approval token.

        Returns:
            Envelope whose data is ``{"job_id": ...}``.
        """

        def work(progress: Progress) -> object:
            return session.run_assembly(request.plan, request.approval_token, progress)

        return _guarded_job(
            session,
            "run_assembly_plan",
            request,
            request.approval_token,
            work,
        )

    @server.tool()
    def get_device_manifest(request: ScenarioInput) -> dict[str, object]:
        """Return sensor and actuator channels for a controller scenario.

        Each channel names its unit (meters, radians, or dimensionless).
        Known scenarios are ``focus_stage``, ``pendulum_release``, and
        ``line_follow``. Fails on an unknown name.
        Example: get_device_manifest(request={"scenario": "focus_stage"}).

        Args:
            request: Scenario name.

        Returns:
            Envelope with ``sensors`` and ``actuators``.
        """
        return _ok(
            session,
            "get_device_manifest",
            request,
            lambda: session.device_manifest(request.scenario),
        )

    @server.tool()
    def check_controller_code(request: CheckCodeInput) -> dict[str, object]:
        """Statically check controller source against the allowlist.

        The source has no units of its own; channels do. A rejected import is
        ``ok: false`` inside the data, not a raised tool error.
        Example: check_controller_code(request={"source": "import os"}).

        Args:
            request: Python source.

        Returns:
            Envelope with the check result.
        """
        return _ok(
            session,
            "check_controller_code",
            request,
            lambda: session.check_code(request.source),
        )

    @server.tool()
    def run_controller(request: RunControllerInput) -> dict[str, object]:
        """Run a controller in the sandbox and return a job id.

        ``duration_s`` is simulated seconds. Channel units come from the
        manifest. The job result is a RunReport: ``summary_text`` plus
        ``plot_images``. A failed check finishes the job with ``ok`` false.
        Real mode requires a token.
        Example: run_controller(request={"source": "import os", "scenario": "focus_stage",
        "duration_s": 0.2}).

        Args:
            request: Source, scenario, duration, params, seed, and token.

        Returns:
            Envelope whose data is ``{"job_id": ...}``.
        """

        def work(progress: Progress) -> object:
            return session.run_controller(request, progress)

        return _guarded_job(session, "run_controller", request, request.approval_token, work)

    @server.tool()
    def validate_protocol(request: ProtocolInput) -> dict[str, object]:
        """Check an experiment protocol against the skill registry.

        Variable units are whatever the protocol names. Unknown skills are
        issues, not exceptions, unless the JSON is not a protocol.
        Example: validate_protocol(request={"protocol": {"id": "p"}}).

        Args:
            request: Protocol object. The token is ignored.

        Returns:
            Envelope with ``valid`` and ``issues``.
        """
        return _ok(
            session,
            "validate_protocol",
            request,
            lambda: session.validate_experiment(request.protocol),
        )

    @server.tool()
    def run_experiment(request: RunExperimentInput) -> dict[str, object]:
        """Run an experiment protocol on a background job.

        Units are the protocol's own. The job result includes ``run_id`` for
        ``fit_models`` and ``get_report``. Real mode requires a token. An
        invalid protocol finishes the job with an error.
        Example: run_experiment(request={"protocol": {"id": "p"}}).

        Args:
            request: Protocol object and approval token.

        Returns:
            Envelope whose data is ``{"job_id": ...}``.
        """

        def work(progress: Progress) -> object:
            return session.run_experiment(request.protocol, request.approval_token, progress)

        return _guarded_job(session, "run_experiment", request, request.approval_token, work)

    @server.tool()
    def fit_models(request: FitModelsInput) -> dict[str, object]:
        """Fit six curves to a finished run and sort them by AIC.

        ``x`` and ``y`` are column names. Their units are the protocol units.
        Lower AIC is a better description of this sample, not a proof.
        Fails when the run or a column is missing.
        Example: fit_models(request={"run_id": "RUN", "x": "concentration_m", "y": "absorbance"}).

        Args:
            request: Run id and column names.

        Returns:
            Envelope with ``fits``.
        """
        return _ok(
            session,
            "fit_models",
            request,
            lambda: session.fit(request.run_id, request.x, request.y),
        )

    @server.tool()
    def get_report(request: GetReportInput) -> dict[str, object]:
        """Write the Markdown report for a run and return its text.

        ``x`` and ``y`` are column names. They are inferred when a successful
        trial has one value and one measurement. The plot is saved next to the
        report. Fails when the run is unknown.
        Example: get_report(request={"run_id": "20260101-000000-abcdef"}).

        Args:
            request: Run id and optional columns.

        Returns:
            Envelope with ``path`` and ``markdown``.
        """
        return _ok(
            session,
            "get_report",
            request,
            lambda: session.write_report(request.run_id, request.x, request.y),
        )

    @server.tool()
    def check_chem_protocol(request: ProtocolInput) -> dict[str, object]:
        """Check a chemistry protocol for volume, tips, and the allowlist.

        Concentrations are molar. Volumes are microliters. Temperatures are
        Celsius. Real mode without a token is a denial in ``safety``, so the
        caller can read the reason. Fails when the JSON is not a protocol.
        Example: check_chem_protocol(request={"protocol": {"id": "dye"}}).

        Args:
            request: Protocol object and optional token.

        Returns:
            Envelope with ``valid``, ``issues``, and ``safety``.
        """
        return _ok(
            session,
            "check_chem_protocol",
            request,
            lambda: session.check_chem(request.protocol, request.approval_token),
        )

    @server.tool()
    def run_chem_protocol(request: RunChemInput) -> dict[str, object]:
        """Run a chemistry protocol on the virtual lab as a job.

        Concentrations are molar and volumes are microliters. The job result
        has a ``run_id`` whose trials can be fit. Real mode requires a token
        and denies a protocol the allowlist rejects.
        Example: run_chem_protocol(request={"protocol": {"id": "dye"}}).

        Args:
            request: Protocol, question, hypothesis, seed, analyte, and token.

        Returns:
            Envelope whose data is ``{"job_id": ...}``.
        """

        def work(progress: Progress) -> object:
            return session.run_chem(request, progress)

        return _guarded_job(
            session,
            "run_chem_protocol",
            request,
            request.approval_token,
            work,
        )

    @server.tool()
    def get_job(request: GetJobInput) -> dict[str, object]:
        """Poll a background job.

        ``progress`` is a fraction from 0 to 1. ``status`` is ``queued``,
        ``running``, ``done``, or ``error``. There is no frame. Fails when
        the id was never issued by this process.
        Example: get_job(request={"job_id": "job-1"}).

        Args:
            request: Job id.

        Returns:
            Envelope with status, progress, result, and error.
        """
        return _ok(session, "get_job", request, lambda: session.jobs.get(request.job_id))


def register_resources(server: MCPServer, session: LabSession) -> None:
    """Attach the physlab resources.

    Args:
        server: MCP server that will publish the resources.
        session: Session that owns the catalog, parts, and run directory.
    """

    @server.resource("physlab://robots", mime_type="application/json")
    def robots() -> str:
        """Return the robot catalog as JSON.

        Returns:
            Catalog rows. ``available`` is false when a mesh is missing.
        """
        return json.dumps(session.catalog_document())

    @server.resource("physlab://parts", mime_type="application/json")
    def parts() -> str:
        """Return the part library as JSON.

        Returns:
            Part names and parameter schemas, millimeters.
        """
        return json.dumps(session.list_parts())

    @server.resource("physlab://skills", mime_type="application/json")
    def skills() -> str:
        """Return experiment, motion, and assembly skill names.

        Returns:
            JSON object with three name lists.
        """
        document = {
            "experiment": list(experiment_registry().names()),
            "motion": list(_MOTION),
            "assembly_ops": list(_ASSEMBLY_OPS),
        }
        return json.dumps(document)

    @server.resource("physlab://runs/{run_id}/report", mime_type="text/markdown")
    def run_report(run_id: str) -> str:
        """Return a Markdown report written by ``get_report``.

        Args:
            run_id: Run id. Must be a single path segment.

        Returns:
            Markdown text.
        """
        return session.report_text(run_id)

    @server.resource("physlab://docs/agent_guide", mime_type="text/markdown")
    def agent_guide() -> str:
        """Return AGENT_GUIDE.md.

        Returns:
            The guide written for an LLM agent.
        """
        return session.agent_guide()


def _params(request: BaseModel | None) -> dict[str, object]:
    if request is None:
        return {}
    return cast(dict[str, object], request.model_dump(mode="json"))


def _ok(
    session: LabSession,
    action: str,
    request: BaseModel | None,
    work: Callable[[], object],
) -> dict[str, object]:
    try:
        envelope = success(work())
    except Exception as exc:
        envelope = failure(exc)
    session.record(action, _params(request), envelope)
    return envelope


def _guarded_job(
    session: LabSession,
    action: str,
    request: BaseModel,
    approval_token: str | None,
    work: Callable[[Progress], object],
) -> dict[str, object]:
    try:
        gate(action, mode=session.config.mode, approval_token=approval_token)
    except Exception as exc:
        envelope = failure(exc)
        session.record(action, _params(request), envelope)
        return envelope
    return _job(session, action, request, work)


def _job(
    session: LabSession,
    action: str,
    request: BaseModel,
    work: Callable[[Progress], object],
) -> dict[str, object]:
    try:
        envelope = success({"job_id": session.jobs.submit(work)})
    except Exception as exc:
        envelope = failure(exc)
    session.record(action, _params(request), envelope)
    return envelope


def _png(
    session: LabSession,
    action: str,
    request: BaseModel,
    work: Callable[[], tuple[dict[str, object], bytes]],
) -> CallToolResult:
    png: bytes | None = None
    try:
        data, png = work()
        envelope = success(data)
        is_error = False
    except Exception as exc:
        envelope = failure(exc)
        is_error = True
    session.record(action, _params(request), envelope)
    blocks: list[ContentBlock] = [TextContent(type="text", text=json.dumps(envelope))]
    if png is not None and not is_error:
        blocks.append(Image(data=png, format="png").to_image_content())
    return CallToolResult(content=blocks, structured_content=envelope, is_error=is_error)
