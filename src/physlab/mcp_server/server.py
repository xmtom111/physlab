"""Build and serve the physlab MCP server.

The installed SDK is mcp 2.3. ``FastMCP`` was renamed to ``MCPServer`` and
lives at ``mcp.server.mcpserver.MCPServer``. This module uses that class.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from physlab.core.config import load_config
from physlab.core.errors import ConfigError
from physlab.core.logging import configure_logging
from physlab.mcp_server.session import LabSession
from physlab.mcp_server.tools import register_resources, register_tools

_INSTRUCTIONS = (
    "Physlab lab server. Read physlab://docs/agent_guide before operating. "
    "Lengths are meters, angles are radians, and quaternions are (w, x, y, z). "
    "CAD parameters are millimeters. Every tool takes a pydantic object in "
    "the request field and returns {ok, data, error}. Calls that can take "
    "longer than two seconds return a job_id; poll get_job. Real mode requires "
    "an approval token from `physlab approve` before hardware or liquid tools."
)


def build_server(session: LabSession) -> MCPServer:
    """Register every tool and resource on a new server.

    Args:
        session: Session the tools close over. One process owns one session.

    Returns:
        Server ready for stdio or streamable HTTP.
    """
    server = MCPServer(name="physlab", instructions=_INSTRUCTIONS)
    register_tools(server, session)
    register_resources(server, session)
    return server


def serve(transport: str, config_path: Path | None = None) -> None:
    """Load config, build one scene, and serve until the client disconnects.

    ``MCP_SCENE`` selects the scene stem. The default is ``table_gantry``.
    The name is not a ``PHYSLAB_`` variable, so it is not merged into the
    config model.

    Args:
        transport: ``stdio`` or ``streamable-http``.
        config_path: Physlab YAML. The default file is used when omitted.

    Raises:
        ConfigError: ``transport`` is not one of the two supported names.
    """
    configure_logging("INFO", False, stream=sys.stderr)
    loaded = load_config(config_path)
    scene_name = os.environ.get("MCP_SCENE", "table_gantry")
    session = LabSession(loaded, scene_name)
    server = build_server(session)
    if transport == "stdio":
        server.run("stdio")
        return
    if transport == "streamable-http":
        server.run("streamable-http")
        return
    raise ConfigError(
        "MCP transport must be stdio or streamable-http",
        details={"transport": transport},
    )
