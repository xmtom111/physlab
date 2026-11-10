"""Design, mock-print, and assemble a lens holder through the MCP server.

The client speaks the MCP stdio transport to ``python -m physlab.mcp_server``.
The scene is ``table_arm``, which needs MuJoCo Menagerie.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import CallToolResult

from physlab.core.config import load_config
from physlab.core.logging import configure_logging, get_logger

_LOG = get_logger("examples.11_mcp_client_demo")
_PLACE = {
    "position_m": [0.46, 0.0, 0.75],
    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    "frame": "world",
}


def main() -> None:
    """Run the scripted client and exit non-zero when assembly misses.

    Raises:
        SystemExit: Menagerie is missing, a tool fails, or the plan misses.
    """
    configure_logging("INFO", False)
    loaded = load_config()
    if not (loaded.assets_dir / "mujoco_menagerie" / ".git").is_dir():
        _LOG.info("assets_missing", hint="physlab assets fetch")
        raise SystemExit(1)
    anyio.run(_run)


async def _run() -> None:
    """Connect over stdio, print a lens holder, and place it."""
    env = dict(os.environ)
    env["MCP_SCENE"] = "table_arm"
    env["MUJOCO_GL"] = "osmesa"
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "physlab.mcp_server"],
        env=env,
    )
    async with (
        stdio_client(parameters) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        designed = _data(
            await session.call_tool(
                "design_part",
                {"request": {"part_name": "lens_holder", "params": {}}},
            )
        )
        printed = _data(
            await session.call_tool(
                "print_part",
                {"request": {"design_id": designed["design_id"]}},
            )
        )
        job = await _poll(session, str(printed["job_id"]))
        if job["status"] != "done":
            raise SystemExit(1)
        item = job["result"]["inventory"]
        plan = {
            "id": "lens_holder_place",
            "components": [
                {
                    "id": "holder",
                    "part_name": "lens_holder",
                    "aruco_id": item["aruco_id"],
                }
            ],
            "steps": [
                {"op": "pick", "component": "holder"},
                {
                    "op": "place",
                    "component": "holder",
                    "target": _PLACE,
                    "tol_pos_m": 0.01,
                    "tol_rot_rad": 0.3,
                },
            ],
        }
        checked = _data(
            await session.call_tool(
                "validate_assembly_plan",
                {"request": {"plan": plan}},
            )
        )
        if not checked["valid"]:
            _LOG.info("plan_rejected", issues=checked["issues"])
            raise SystemExit(1)
        started = _data(await session.call_tool("run_assembly_plan", {"request": {"plan": plan}}))
        finished = await _poll(session, str(started["job_id"]), attempts=600)
        if finished["status"] != "done" or not finished["result"]["success"]:
            _LOG.info("assembly_failed", job=finished)
            raise SystemExit(1)
        _LOG.info("assembled", run=finished["result"])


async def _poll(session: ClientSession, job_id: str, attempts: int = 50) -> dict[str, Any]:
    """Wait until ``job_id`` is done or has failed.

    Args:
        session: Open MCP session.
        job_id: Id returned by a long-running tool.
        attempts: How many polls to try. Each wait is 0.5 seconds.

    Returns:
        The job snapshot.

    Raises:
        SystemExit: The job did not finish in time.
    """
    snapshot: dict[str, Any] = {}
    for _ in range(attempts):
        snapshot = _data(await session.call_tool("get_job", {"request": {"job_id": job_id}}))
        if snapshot["status"] in {"done", "error"}:
            return snapshot
        await anyio.sleep(0.5)
    raise SystemExit(1)


def _data(result: CallToolResult) -> dict[str, Any]:
    """Return the envelope data or exit when the tool failed.

    Args:
        result: Tool result from the MCP client.

    Returns:
        The ``data`` object.

    Raises:
        SystemExit: The envelope is not ok.
    """
    body = result.structured_content
    if not isinstance(body, dict):
        text = next(block.text for block in result.content if block.type == "text")
        loaded = json.loads(text)
        body = loaded if isinstance(loaded, dict) else {}
    if not body.get("ok"):
        _LOG.info("tool_failed", error=body.get("error"))
        raise SystemExit(1)
    data = body["data"]
    if not isinstance(data, dict):
        raise SystemExit(1)
    return data


if __name__ == "__main__":
    main()
