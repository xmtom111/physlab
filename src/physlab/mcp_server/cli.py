"""``physlab mcp serve`` and ``physlab approve``."""

from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path
from typing import Annotated

import typer

from physlab.core.config import load_config
from physlab.core.errors import PhysLabError
from physlab.mcp_server.server import serve
from physlab.safety.gate import assert_not_protected

mcp_app = typer.Typer(add_completion=False, no_args_is_help=True)
_TRANSPORTS = ("stdio", "streamable-http")


def register(app: typer.Typer) -> None:
    """Attach the MCP server command and the human approval command.

    Args:
        app: Root physlab Typer application.
    """
    app.add_typer(mcp_app, name="mcp")
    app.command("approve")(approve)


@mcp_app.callback()
def mcp_main() -> None:
    """Serve one physlab session to an MCP client."""


@mcp_app.command("serve")
def mcp_serve(
    transport: Annotated[
        str,
        typer.Option("--transport", help="stdio or streamable-http."),
    ] = "stdio",
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Physlab YAML config. Defaults to configs/default.yaml."),
    ] = None,
) -> None:
    """Serve one scene. MCP_SCENE selects the scene stem."""
    if transport not in _TRANSPORTS:
        typer.echo("transport must be stdio or streamable-http", err=True)
        raise typer.Exit(code=2)
    try:
        serve(transport, config)
    except PhysLabError as exc:
        typer.echo(exc.message, err=True)
        raise typer.Exit(code=1) from exc


def approve(
    plan_or_protocol_file: Annotated[
        Path,
        typer.Argument(
            help="Plan or protocol file a human is approving.",
            exists=True,
            dir_okay=False,
        ),
    ],
) -> None:
    """Mint an approval token. T12 verifies it. This command only records it.

    The token is printed on stdout. A copy is written under the run directory.
    Protected chemistry and safety configs are refused.
    """
    loaded = load_config()
    try:
        assert_not_protected(plan_or_protocol_file, loaded.configs_dir)
    except PhysLabError as exc:
        typer.echo(exc.message, err=True)
        raise typer.Exit(code=1) from exc
    payload = plan_or_protocol_file.read_bytes()
    token = secrets.token_hex(16)
    record = {
        "token": token,
        "file": str(plan_or_protocol_file.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    directory = loaded.run_dir / "approvals"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{token}.json"
    destination.write_text(json.dumps(record, indent=2), encoding="utf-8")
    typer.echo(token)
