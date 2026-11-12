"""In-process MCP client coverage for the physlab server."""

from __future__ import annotations

import json
import time
from pathlib import Path

import anyio
import pytest
from mcp.client.session import ClientSession
from mcp.shared.message import SessionMessage
from mcp.types import CallToolResult

from physlab.chem.workflows import dye_calibration_protocol
from physlab.core.config import load_config
from physlab.core.events import Actor
from physlab.core.types import Mode
from physlab.mcp_server.jobs import JobManager
from physlab.mcp_server.scenes import list_scene_names, load_scene_file, scene_path
from physlab.mcp_server.server import build_server
from physlab.mcp_server.session import LabSession
from physlab.safety.gate import assert_not_protected, gate

_TOOLS = {
    "get_session_info",
    "reset_scene",
    "list_scenes",
    "load_scene",
    "list_robots",
    "get_robot_state",
    "move_arm_joints",
    "move_arm_pose",
    "gripper",
    "move_base",
    "stop_all",
    "capture_image",
    "locate_marker",
    "list_parts",
    "design_part",
    "check_part",
    "print_part",
    "list_inventory",
    "validate_assembly_plan",
    "run_assembly_plan",
    "get_device_manifest",
    "check_controller_code",
    "run_controller",
    "validate_protocol",
    "run_experiment",
    "fit_models",
    "get_report",
    "check_chem_protocol",
    "run_chem_protocol",
    "get_job",
}


def test_gate_requires_a_token_only_in_real_mode() -> None:
    gate("move_arm_joints", mode=Mode.SIM, approval_token=None)
    gate("move_arm_joints", mode=Mode.REAL, approval_token="abc")
    with pytest.raises(Exception, match="approval token"):
        gate("move_arm_joints", mode=Mode.REAL, approval_token="  ")


def test_protected_config_paths_are_refused(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    secret = configs / "safety" / "real.yaml"
    secret.parent.mkdir(parents=True)
    secret.write_text("{}\n", encoding="utf-8")
    with pytest.raises(Exception, match="Protected"):
        assert_not_protected(secret, configs)
    assert_not_protected(tmp_path / "plan.yaml", configs)


def test_scene_files_list_and_parse() -> None:
    loaded = load_config()
    names = list_scene_names(loaded.configs_dir)
    assert "table_gantry" in names
    assert "table_arm" in names
    parsed, scene = load_scene_file(scene_path(loaded.configs_dir, "table_gantry"))
    assert parsed == "table_gantry"
    assert scene.robots[0].instance == "gantry"
    assert {camera.name for camera in scene.cameras} == {"overhead_left", "overhead_right"}


def test_job_records_an_error() -> None:
    jobs = JobManager()

    def boom(_progress: object) -> object:
        raise RuntimeError("nope")

    job_id = jobs.submit(boom)
    snapshot: dict[str, object] = {}
    for _ in range(50):
        snapshot = jobs.get(job_id)
        if snapshot["status"] == "error":
            break
        time.sleep(0.01)
    assert snapshot["status"] == "error"
    error = snapshot["error"]
    assert isinstance(error, dict)
    assert error["type"] == "RuntimeError"
    with pytest.raises(Exception, match="Unknown job"):
        jobs.get("missing")


@pytest.mark.slow
def test_server_tools_jobs_images_and_real_mode_gate(tmp_path: Path) -> None:
    loaded = load_config().model_copy(update={"run_dir": tmp_path, "seed": 1})
    session = LabSession(loaded, "table_gantry")
    server = build_server(session)
    anyio.run(_exercise, server)
    real = loaded.model_copy(update={"mode": Mode.REAL, "run_dir": tmp_path / "real"})
    real_session = LabSession(real, "table_gantry")
    real_server = build_server(real_session)
    anyio.run(_reject_real, real_server)
    events = session.event_log.read_all()
    assert any(event.actor is Actor.AGENT and event.action == "list_robots" for event in events)


async def _exercise(server: object) -> None:
    from mcp.server.mcpserver import MCPServer

    assert isinstance(server, MCPServer)
    client_send, server_read = anyio.create_memory_object_stream[SessionMessage](20)
    server_send, client_read = anyio.create_memory_object_stream[SessionMessage | Exception](20)
    options = server._lowlevel_server.create_initialization_options()

    async def serve() -> None:
        await server._lowlevel_server.run(server_read, server_send, options, raise_exceptions=True)

    async def client() -> None:
        async with ClientSession(client_read, client_send) as mcp:
            await mcp.initialize()
            listed = await mcp.list_tools()
            assert {tool.name for tool in listed.tools} == _TOOLS
            info = await _data(mcp, "get_session_info")
            assert info["scene"] == "table_gantry"
            assert info["mode"] == "sim"
            scenes = await _data(mcp, "list_scenes")
            assert "table_gantry" in scenes["scenes"]
            reset = await _data(mcp, "reset_scene", {"request": {"seed": 1}})
            assert reset["seed"] == 1
            robots = await _data(mcp, "list_robots")
            assert robots["robots"][0]["instance"] == "gantry"
            state = await _data(mcp, "get_robot_state", {"request": {"instance": "gantry"}})
            assert state["joints"]["positions_rad"] == [0.0, 0.0, 0.0]
            joint_target = {"instance": "gantry", "q_rad": [0.05, 0.0, 0.0], "speed_scale": 1.0}
            moved = await _data(mcp, "move_arm_joints", {"request": joint_target})
            assert moved["success"] is True
            grip_request = {"instance": "gantry", "action": "open"}
            grip = await _envelope(mcp, "gripper", {"request": grip_request})
            assert grip["ok"] is False
            base = await _envelope(
                mcp,
                "move_base",
                {"request": {"instance": "gantry", "x_m": 0.1, "y_m": 0.0, "yaw_rad": 0.0}},
            )
            assert base["ok"] is False
            stopped = await _data(mcp, "stop_all", {"request": {}})
            assert "gantry.arm" in stopped["stopped"]
            shot = await mcp.call_tool(
                "capture_image",
                {"request": {"camera": "overhead_left"}},
            )
            images = [block for block in shot.content if block.type == "image"]
            assert images and images[0].mime_type == "image/png"
            image_body = _body(shot)
            assert image_body["ok"] is True
            assert image_body["data"]["width"] == 320
            missing = await _data(mcp, "locate_marker", {"request": {"marker_id": 3}})
            assert missing["found"] is False
            parts = await _data(mcp, "list_parts")
            assert any(part["name"] == "led_holder" for part in parts["parts"])
            designed = await mcp.call_tool(
                "design_part",
                {"request": {"part_name": "led_holder", "params": {}}},
            )
            assert any(block.type == "image" for block in designed.content)
            design = _body(designed)["data"]
            assert design["preview_png_base64"]
            design_request = {"design_id": design["design_id"]}
            checked = await _data(mcp, "check_part", {"request": design_request})
            assert "ok" in checked
            printed = await _data(mcp, "print_part", {"request": design_request})
            job = await _poll(mcp, printed["job_id"])
            assert job["status"] == "done"
            assert job["result"]["inventory"]["part_name"] == "led_holder"
            stock = await _data(mcp, "list_inventory")
            assert stock["items"][0]["aruco_id"] is not None
            plan = {"id": "empty", "components": [], "steps": []}
            issues = await _data(mcp, "validate_assembly_plan", {"request": {"plan": plan}})
            assert issues["valid"] is True
            assembly = await _data(mcp, "run_assembly_plan", {"request": {"plan": plan}})
            assembly_job = await _poll(mcp, assembly["job_id"])
            assert assembly_job["status"] == "error"
            manifest = await _data(
                mcp,
                "get_device_manifest",
                {"request": {"scenario": "focus_stage"}},
            )
            assert "defocus_m" in manifest["sensors"]
            code = await _data(mcp, "check_controller_code", {"request": {"source": "import os\n"}})
            assert code["ok"] is False
            controller = await _data(
                mcp,
                "run_controller",
                {
                    "request": {
                        "source": "import os\n",
                        "scenario": "focus_stage",
                        "duration_s": 0.2,
                    }
                },
            )
            control_job = await _poll(mcp, controller["job_id"])
            assert control_job["status"] == "done"
            assert control_job["result"]["ok"] is False
            assert control_job["result"]["phase_failed"] == "CHECK"
            assert "CHECK" in control_job["result"]["summary_text"]
            bad = {
                "id": "nope",
                "question": "q",
                "hypothesis": "h",
                "apparatus": "bench",
                "dependent": [{"name": "y", "unit": "1", "method": "read"}],
                "procedure": [{"skill": "not_a_skill"}],
                "repetitions": 1,
                "seed": 0,
                "safety_class": "sim_only",
            }
            invalid = await _data(mcp, "validate_protocol", {"request": {"protocol": bad}})
            assert invalid["valid"] is False
            experiment = await _data(mcp, "run_experiment", {"request": {"protocol": bad}})
            experiment_job = await _poll(mcp, experiment["job_id"])
            assert experiment_job["status"] == "error"
            protocol = dye_calibration_protocol().model_dump(mode="json")
            chem_check = await _data(
                mcp,
                "check_chem_protocol",
                {"request": {"protocol": protocol}},
            )
            assert chem_check["valid"] is True
            chem = await _data(
                mcp,
                "run_chem_protocol",
                {"request": {"protocol": protocol, "question": "q", "hypothesis": "h"}},
            )
            chem_job = await _poll(mcp, chem["job_id"])
            assert chem_job["status"] == "done"
            run_id = chem_job["result"]["run_id"]
            fits = await _data(
                mcp,
                "fit_models",
                {"request": {"run_id": run_id, "x": "concentration_m", "y": "absorbance"}},
            )
            assert fits["fits"]
            report = await _data(mcp, "get_report", {"request": {"run_id": run_id}})
            assert "absorbance" in report["markdown"]
            resources = await mcp.list_resources()
            uris = {str(item.uri) for item in resources.resources}
            assert "physlab://robots" in uris
            assert "physlab://parts" in uris
            assert "physlab://skills" in uris
            assert "physlab://docs/agent_guide" in uris
            guide = await mcp.read_resource("physlab://docs/agent_guide")
            assert "approval" in guide.contents[0].text.lower()
            written = await mcp.read_resource(f"physlab://runs/{run_id}/report")
            assert "absorbance" in written.contents[0].text
            reloaded = await _data(mcp, "load_scene", {"request": {"name": "table_gantry"}})
            assert reloaded["scene"] == "table_gantry"
        await client_send.aclose()

    async with anyio.create_task_group() as group:
        group.start_soon(serve)
        group.start_soon(client)


async def _reject_real(server: object) -> None:
    from mcp.server.mcpserver import MCPServer

    assert isinstance(server, MCPServer)
    client_send, server_read = anyio.create_memory_object_stream[SessionMessage](8)
    server_send, client_read = anyio.create_memory_object_stream[SessionMessage | Exception](8)
    options = server._lowlevel_server.create_initialization_options()

    async def serve() -> None:
        await server._lowlevel_server.run(server_read, server_send, options, raise_exceptions=True)

    async def client() -> None:
        async with ClientSession(client_read, client_send) as mcp:
            await mcp.initialize()
            denied = await _envelope(
                mcp,
                "move_arm_joints",
                {"request": {"instance": "gantry", "q_rad": [0.0, 0.0, 0.0]}},
            )
            assert denied["ok"] is False
            assert denied["error"]["type"] == "ApprovalRequired"
            chem = await _envelope(
                mcp,
                "run_chem_protocol",
                {"request": {"protocol": dye_calibration_protocol().model_dump(mode="json")}},
            )
            assert chem["ok"] is False
            assert chem["error"]["type"] == "ApprovalRequired"
        await client_send.aclose()

    async with anyio.create_task_group() as group:
        group.start_soon(serve)
        group.start_soon(client)


async def _envelope(
    mcp: ClientSession,
    name: str,
    arguments: dict[str, object] | None = None,
) -> dict[str, object]:
    result = await mcp.call_tool(name, arguments)
    return _body(result)


async def _data(
    mcp: ClientSession,
    name: str,
    arguments: dict[str, object] | None = None,
) -> dict[str, object]:
    body = await _envelope(mcp, name, arguments)
    assert body["ok"] is True, body
    data = body["data"]
    assert isinstance(data, dict)
    return data


async def _poll(mcp: ClientSession, job_id: object) -> dict[str, object]:
    assert isinstance(job_id, str)
    snapshot: dict[str, object] = {}
    for _ in range(100):
        snapshot = await _data(mcp, "get_job", {"request": {"job_id": job_id}})
        if snapshot["status"] in {"done", "error"}:
            return snapshot
        await anyio.sleep(0.05)
    raise AssertionError(snapshot)


def _body(result: CallToolResult) -> dict[str, object]:
    if isinstance(result.structured_content, dict):
        return result.structured_content
    text = next(block.text for block in result.content if block.type == "text")
    loaded = json.loads(text)
    assert isinstance(loaded, dict)
    return loaded
