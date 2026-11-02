"""Stdio MCP printer with tool names taken from the environment.

The production client is pointed at this process by a test YAML file. Tool
names are not imported from physlab.
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer


def main() -> None:
    """Register whichever roles the environment names, then serve stdio."""
    server = MCPServer("fake-printer")
    jobs: dict[str, str] = {}
    names = {
        "slice": os.environ.get("FAKE_SLICE", ""),
        "upload": os.environ.get("FAKE_UPLOAD", ""),
        "start": os.environ.get("FAKE_START", ""),
        "status": os.environ.get("FAKE_STATUS", ""),
        "cancel": os.environ.get("FAKE_CANCEL", ""),
    }
    log_path = os.environ.get("FAKE_LOG", "")

    def note(role: str) -> None:
        if not log_path:
            return
        with Path(log_path).open("a", encoding="utf-8") as handle:
            handle.write(names[role] + "\n")

    if names["slice"]:

        @server.tool(name=names["slice"])
        def slice_mesh(mesh_path: str, part: str) -> dict[str, object]:
            note("slice")
            return {"mesh_path": mesh_path, "part": part}

    if names["upload"]:

        @server.tool(name=names["upload"])
        def upload_mesh(
            mesh_path: str,
            filament: str,
            n_copies: int,
            part: str,
            quality: str,
        ) -> dict[str, object]:
            note("upload")
            token = f"job-{len(jobs) + 1}"
            jobs[token] = "DONE"
            return {
                "token": token,
                "phase": "QUEUED",
                "fraction": 0.0,
                "note": f"queued {part} {filament} {quality} x{n_copies} {mesh_path}",
                "remaining_s": 1.0,
            }

    if names["start"]:

        @server.tool(name=names["start"])
        def begin(id: str) -> dict[str, object]:
            note("start")
            return {
                "token": id,
                "phase": "PRINTING",
                "fraction": 0.1,
                "note": "started",
                "remaining_s": 1.0,
            }

    if names["status"]:

        @server.tool(name=names["status"])
        def job_state(id: str) -> dict[str, object]:
            note("status")
            phase = jobs.get(id, "FAILED")
            return {
                "token": id,
                "phase": phase,
                "fraction": 1.0 if phase == "DONE" else 0.0,
                "note": phase.lower(),
                "remaining_s": 0.0,
            }

    if names["cancel"]:

        @server.tool(name=names["cancel"])
        def stop_job(id: str) -> dict[str, object]:
            note("cancel")
            jobs[id] = "CANCELLED"
            return {
                "token": id,
                "phase": "CANCELLED",
                "fraction": 0.0,
                "note": "cancelled",
                "remaining_s": 0.0,
            }

    server.run("stdio")


if __name__ == "__main__":
    main()
