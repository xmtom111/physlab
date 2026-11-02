"""Part library, print checks, mock spawn, and the MCP print client."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import pytest
import yaml
from typer.testing import CliRunner

from physlab.cli import app
from physlab.core.config import load_config
from physlab.core.errors import ConfigError
from physlab.core.types import Pose
from physlab.fabrication.cad import build_part, list_parts
from physlab.fabrication.mcp_print import McpPrintService
from physlab.fabrication.mesh_check import PrinterProfile, check_printable, load_printer_profile
from physlab.fabrication.mock_print import MockPrintService
from physlab.fabrication.models import PrintJob, PrintState
from physlab.fabrication.print_service import fabricate
from physlab.hal.factory import create_robot
from physlab.robots.catalog import RobotCatalog
from physlab.sim.scene import RobotPlacement, SceneSpec, TableSpec
from physlab.sim.world import SimWorld

_BED = PrinterProfile(bed_size_mm=(220, 220), max_height_mm=250, nozzle_mm=0.4, layer_mm=0.2)
_SERVER = Path(__file__).with_name("fake_mcp_printer.py")
_RENAMED = {
    "slice": "cut_model",
    "upload": "send_mesh",
    "start": "begin_job",
    "status": "job_state",
    "cancel": "halt_job",
}


def test_list_parts_publishes_json_schemas() -> None:
    listed = list_parts()
    names = [item.name for item in listed]
    assert names == [
        "magnetic_base",
        "lens_holder",
        "tube",
        "sample_stage",
        "camera_mount",
        "led_holder",
        "rail_baseplate",
        "pendulum_kit",
        "incline_ramp",
    ]
    lens = next(item for item in listed if item.name == "lens_holder")
    properties = lens.json_schema["properties"]
    assert isinstance(properties, dict)
    assert "lens_d_mm" in properties


@pytest.mark.parametrize(
    "name",
    [
        "magnetic_base",
        "lens_holder",
        "tube",
        "sample_stage",
        "camera_mount",
        "led_holder",
        "rail_baseplate",
        "pendulum_kit",
        "incline_ramp",
    ],
)
def test_default_part_is_printable(name: str, tmp_path: Path) -> None:
    artifact = build_part(name, directory=tmp_path)
    assert artifact.stl_path.is_file()
    assert artifact.step_path.is_file()
    assert "grasp" in artifact.frames
    assert "mount" in artifact.frames
    assert artifact.aruco_face is not None
    report = check_printable(artifact.stl_path, _BED)
    assert report.watertight
    assert report.fits_bed
    assert report.ok, report.issues
    assert report.volume_cm3 > 0


def test_oversized_tube_does_not_fit_the_bed(tmp_path: Path) -> None:
    artifact = build_part("tube", {"length_mm": 300, "inner_d_mm": 20, "wall_mm": 3}, tmp_path)
    report = check_printable(artifact.stl_path, _BED)
    assert report.fits_bed is False
    assert report.ok is False
    assert any("fit the bed" in issue for issue in report.issues)


def test_profile_yaml_matches_the_220_bed() -> None:
    profile = load_printer_profile()
    assert profile.bed_size_mm == (220, 220)
    assert profile.max_height_mm == 250


def test_fabricate_stops_before_submit_when_the_part_does_not_fit() -> None:
    class _Unused:
        def submit(self, job: object) -> str:
            del job
            raise AssertionError("submit should not be called")

        def status(self, job_id: str) -> object:
            del job_id
            raise AssertionError

        def cancel(self, job_id: str) -> None:
            del job_id

    result = fabricate(
        "tube",
        {"length_mm": 300, "inner_d_mm": 20, "wall_mm": 3},
        _Unused(),  # type: ignore[arg-type]
        _BED,
    )
    assert result.ok is False
    assert result.job_id is None
    assert result.issues


def test_mock_clock_reaches_done_and_records_inventory(tmp_path: Path) -> None:
    artifact = build_part("led_holder", directory=tmp_path)
    clock = _Clock()
    service = MockPrintService(None, mm3_per_s=1000.0, clock=clock)
    job_id = service.submit(PrintJob(part=artifact))
    assert service.status(job_id).state is PrintState.QUEUED
    mesh_volume = _volume(artifact.stl_path)
    clock.now = mesh_volume / 1000.0
    done = service.status(job_id)
    assert done.state is PrintState.DONE
    item = service.inventory_entry(job_id)
    assert item is not None
    assert item.part_name == "led_holder"
    assert item.location == "virtual"
    assert item.aruco_id is None


def test_mock_cancel_stops_a_queued_job(tmp_path: Path) -> None:
    artifact = build_part("led_holder", directory=tmp_path)
    service = MockPrintService(None, mm3_per_s=1.0, clock=_Clock())
    job_id = service.submit(PrintJob(part=artifact))
    service.cancel(job_id)
    assert service.status(job_id).state is PrintState.CANCELLED


def test_unknown_part_and_job_are_config_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Unknown part"):
        build_part("not_a_part", directory=tmp_path)
    service = MockPrintService(None, clock=_Clock())
    with pytest.raises(ConfigError, match="Unknown print job"):
        service.status("missing")


@pytest.mark.assets
def test_mock_spawn_is_picked_by_the_xarm(tmp_path: Path) -> None:
    loaded = load_config()
    config = loaded.model_copy(update={"sim": loaded.sim.model_copy(update={"timestep_s": 0.002})})
    scene = SceneSpec(
        table=TableSpec(),
        robots=[
            RobotPlacement(
                robot_id="ufactory_xarm7",
                instance="arm",
                base_pose=Pose(position_m=(0.0, 0.0, 0.75), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
            )
        ],
    )
    world = SimWorld(scene, RobotCatalog(config.configs_dir, config.assets_dir), config)
    robot = create_robot(world, "arm")
    assert robot.arm is not None and robot.gripper is not None
    slot = Pose(position_m=(0.48, 0.0, 0.75), quat_wxyz=(1.0, 0.0, 0.0, 0.0))
    service = MockPrintService(
        world,
        mm3_per_s=1.0e9,
        pickup_poses=[slot],
    )
    result = fabricate("lens_holder", {}, service, _BED)
    assert result.ok
    item = result.inventory
    assert item is not None
    assert item.location == "print_pickup_zone"
    assert item.aruco_id == 31
    assert item.pose is not None
    body = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, item.part_instance_id)
    assert body >= 0
    grasp = item.pose.compose(result.part.frames["grasp"]) if result.part is not None else slot
    assert robot.arm.move_pose(grasp, speed_scale=1.0, timeout_s=12.0).success
    assert robot.gripper.close().success
    height = grasp.position_m[2] + 0.08
    lifted = grasp.model_copy(
        update={"position_m": (grasp.position_m[0], grasp.position_m[1], height)}
    )
    assert robot.arm.move_pose(lifted, speed_scale=1.0, timeout_s=12.0).success
    height = float(world.data.xpos[body][2])
    assert height > 0.78
    del tmp_path


def test_mcp_renamed_tools_submit_and_missing_tool_lists_server_names(tmp_path: Path) -> None:
    log_path = tmp_path / "calls.log"
    config_path = _mapping(tmp_path, _RENAMED, log_path)
    service = McpPrintService(config_path)
    try:
        schemas = service.connect()
        assert set(schemas) == set(_RENAMED.values())
        artifact = build_part("magnetic_base", directory=tmp_path)
        job_id = service.submit(PrintJob(part=artifact))
        assert job_id == "job-1"
        assert service.status(job_id).state is PrintState.DONE
        service.cancel(job_id)
        entry = service.inventory_entry(job_id)
        assert entry is not None
        assert entry.location == "printer"
    finally:
        service.close()
    calls = log_path.read_text(encoding="utf-8")
    assert "send_mesh" in calls
    assert "begin_job" in calls
    assert "upload" not in calls

    broken = dict(_RENAMED)
    broken["upload"] = "not_a_tool"
    bad = McpPrintService(_mapping(tmp_path, broken, None, _RENAMED, name="broken.yaml"))
    try:
        with pytest.raises(ConfigError, match="send_mesh") as caught:
            bad.connect()
        assert "not_a_tool" in str(caught.value.details.get("missing"))
    finally:
        bad.close()


def test_printer_cli_lists_tools_and_submits(tmp_path: Path) -> None:
    config_path = _mapping(tmp_path, _RENAMED, tmp_path / "cli.log")
    runner = CliRunner()
    listed = runner.invoke(app, ["printer", "tools", "--config", str(config_path)])
    assert listed.exit_code == 0, listed.stderr
    assert "send_mesh" in listed.stdout
    assert "begin_job" in listed.stdout
    submitted = runner.invoke(
        app,
        ["printer", "submit", "led_holder", "--params", "{}", "--config", str(config_path)],
    )
    assert submitted.exit_code == 0, submitted.stderr
    assert submitted.stdout.strip() == "job-1"


def _mapping(
    directory: Path,
    tools: dict[str, str],
    log_path: Path | None,
    server_tools: dict[str, str] | None = None,
    name: str = "printer.yaml",
) -> Path:
    exposed = server_tools or tools
    env = {"FAKE_LOG": "" if log_path is None else str(log_path)}
    for role, tool_name in exposed.items():
        env[f"FAKE_{role.upper()}"] = tool_name
    document = {
        "server": {"command": sys.executable, "args": [str(_SERVER)], "env": env},
        "tools": tools,
        "arg_templates": {
            "slice": {"stl_path": "mesh_path", "part_name": "part"},
            "upload": {
                "stl_path": "mesh_path",
                "material": "filament",
                "copies": "n_copies",
                "part_name": "part",
                "quality": "quality",
            },
            "start": {"job_id": "id"},
            "status": {"job_id": "id"},
            "cancel": {"job_id": "id"},
        },
        "result_fields": {
            "job_id": "token",
            "state": "phase",
            "progress": "fraction",
            "message": "note",
            "eta_s": "remaining_s",
        },
    }
    path = directory / name
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _volume(path: Path) -> float:
    import trimesh

    mesh = trimesh.load(path, force="mesh")
    return abs(float(mesh.volume))


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now
