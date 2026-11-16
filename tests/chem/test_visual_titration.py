"""Persistent microprotocol and headless visual bridge regressions."""

from pathlib import Path
from unittest.mock import patch

import anyio
import httpx
import mujoco
import numpy as np
import pytest

from physlab.chem.demo_gif import write_demo_gif
from physlab.chem.gantry_bridge import liquid_height, liquid_rgba, well_position
from physlab.chem.llm_agent import FakeLLM, HttpLLM, run_agent
from physlab.chem.virtual_lab import VirtualChemLab
from physlab.chem.visual_titration import dose_protocol, initial_protocol, unsafe_protocols
from physlab.core.config import load_config
from physlab.core.errors import ConfigError
from physlab.mcp_server.models import RunChemInput
from physlab.mcp_server.session import LabSession


def test_incremental_and_default_reset() -> None:
    lab = VirtualChemLab()
    lab.execute(initial_protocol())
    low = lab.execute(dose_protocol(10), reset=False)[0]
    for _ in range(3):
        lab.execute(dose_protocol(10), reset=False)
    high = lab.execute(dose_protocol(2), reset=False)[0]
    assert high.measurements["ph"] > low.measurements["ph"] + 4
    assert lab._wells["2:A1"].volume_ul == 132
    assert high.values["titrant_ul"] == 42
    with pytest.raises(ConfigError):
        lab.execute(initial_protocol(), reset=False)
    # A rejected request must leave both state and history unchanged.
    assert lab._wells["2:A1"].volume_ul == 132
    lab.execute(initial_protocol())
    assert lab._wells["2:A1"].volume_ul == 90
    with pytest.raises(ValueError):
        dose_protocol(200)


def test_colors_and_heights() -> None:
    assert liquid_rgba(5)[3] < 0.21
    assert liquid_rgba(10)[1] < 0.3
    assert liquid_rgba(10)[3] > 0.8
    assert liquid_rgba(7, 2, False)[3] > liquid_rgba(7, 0.01, False)[3]
    assert liquid_rgba(7, 2, False)[1] < liquid_rgba(7, 0.01, False)[1]
    assert liquid_height(0) < liquid_height(300)
    assert liquid_height(300) == pytest.approx(0.92 * 0.012)
    # Independent channel transmittances multiply for mixed indicator/dye.
    assert np.all(liquid_rgba(10, 2)[1:3] <= liquid_rgba(10, 0)[1:3])
    for slot in ("1", "2", "3"):
        for well in ("A1", "H12"):
            x, y, z = well_position(f"{slot}:{well}")
            assert 0 <= x <= 0.5 and 0 <= y <= 0.5 and 0.75 <= z <= 1.2


def test_session_checks_and_bridge(tmp_path: Path) -> None:
    session = LabSession(load_config().model_copy(update={"run_dir": tmp_path}), "chem_gantry")
    assert session.world.model.njnt == 3
    assert np.all(session.world.model.jnt_type == mujoco.mjtJoint.mjJNT_SLIDE)
    for name, protocol in unsafe_protocols().items():
        with patch.object(
            VirtualChemLab, "execute", side_effect=AssertionError("must not execute")
        ):
            result = session.check_chem(protocol.model_dump(mode="json"), None)
        assert not result["valid"], name
        assert result["issues"] or result["safety"]["violations"]
    session.run_chem(
        RunChemInput(protocol=initial_protocol().model_dump(mode="json")), lambda _: None
    )
    snapshot = session._chem_lab._wells["2:A1"].volume_ul
    for _ in range(2):
        assert session.check_chem(dose_protocol(10).model_dump(mode="json"), None)["valid"]
    assert session._chem_lab._wells["2:A1"].volume_ul == snapshot
    for _ in range(2):
        session.run_chem(
            RunChemInput(protocol=dose_protocol(10).model_dump(mode="json")), lambda _: None
        )
    assert session._chem_lab._wells["2:A1"].volume_ul == 110
    model = session.world.model
    gid = model.geom("liquid_2:A1").id
    assert model.geom_size[gid, 1] == pytest.approx(liquid_height(110) / 2)
    assert model.geom_rgba[gid, 3] > 0
    assert session.chem_bridge.frames
    session.reset(None, None)
    assert session._chem_lab is None
    assert model.geom_rgba[gid, 3] == 0
    session.load("table_gantry")
    assert session.chem_bridge is None


def test_http_missing_key_and_bounded_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_DEMO_LLM_KEY", raising=False)
    with pytest.raises(ValueError, match="TEST_DEMO_LLM_KEY"):
        HttpLLM("fake", key_env="TEST_DEMO_LLM_KEY")
    # Dummy value is local to this test; no real credentials are read.
    monkeypatch.setenv("TEST_DEMO_LLM_KEY", "test-placeholder")
    client = HttpLLM("test", "https://example.invalid/v1", "TEST_DEMO_LLM_KEY")
    reply = httpx.Response(
        200,
        json={"choices": [{"message": {"content": '{"dose_ul": 2}'}}]},
        request=httpx.Request("POST", "https://example.invalid"),
    )
    with patch("httpx.post", return_value=reply) as post:
        assert client.choose("titrate", [(30, 7.6)]) == 2
        assert post.call_args.args[0] == "https://example.invalid/v1/chat/completions"
    reply = httpx.Response(
        200,
        json={"choices": [{"message": {"content": '{"dose_ul": 500}'}}]},
        request=httpx.Request("POST", "https://example.invalid"),
    )
    with patch("httpx.post", return_value=reply), pytest.raises(ValueError, match="2 or 10"):
        client.choose("titrate", [(30, 7.6)])


def test_offline_agent_and_gif(tmp_path: Path) -> None:
    session = LabSession(load_config().model_copy(update={"run_dir": tmp_path}), "chem_gantry")
    with patch("httpx.post", side_effect=AssertionError("offline")):
        result = anyio.run(run_agent, session, FakeLLM())
    assert result["observations"][-1][0] > 40
    assert result["observations"][-1][1] > 9.5
    doses = np.diff([point[0] for point in result["observations"]])
    assert 10 in doses and 2 in doses
    assert sum(line.startswith("BLOCKED") for line in result["logs"]) == 4
    assert Path(result["report"]["path"]).is_file()
    assert result["fits"]["fits"]
    path = tmp_path / "demo.gif"
    write_demo_gif(result["frames"][::20], path)
    assert path.read_bytes().startswith(b"GIF")
