"""``physlab printer`` commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from physlab.core.errors import PhysLabError
from physlab.fabrication.mcp_print import McpPrintService
from physlab.fabrication.mesh_check import load_printer_profile
from physlab.fabrication.print_service import fabricate

printer_app = typer.Typer(add_completion=False, no_args_is_help=True)


def register(app: typer.Typer) -> None:
    """Attach the printer command group.

    Args:
        app: Root physlab Typer application.
    """
    app.add_typer(printer_app, name="printer")


@printer_app.callback()
def printer_main() -> None:
    """Talk to the printer MCP server described by the YAML mapping."""


@printer_app.command("tools")
def printer_tools(
    config: Annotated[
        Path | None,
        typer.Option(
            "--config", help="Printer MCP YAML. Defaults to configs/printers/printer_mcp.yaml."
        ),
    ] = None,
) -> None:
    """Print the server's tool names and input schemas."""
    service = McpPrintService(config)
    try:
        tools = service.connect()
    except PhysLabError as exc:
        _echo_error(exc)
        raise typer.Exit(code=1) from exc
    finally:
        service.close()
    for name in sorted(tools):
        typer.echo(f"{name}\t{json.dumps(tools[name], sort_keys=True)}")


@printer_app.command("submit")
def printer_submit(
    part: Annotated[str, typer.Argument(help="Part library name.")],
    params: Annotated[str, typer.Option("--params", help="JSON object of part parameters.")] = "{}",
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Printer MCP YAML."),
    ] = None,
) -> None:
    """Build a part, check it, and submit it through the mapped MCP tools."""
    try:
        overrides = _parse_params(params)
    except json.JSONDecodeError as exc:
        typer.echo(f"params are not JSON: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    service = McpPrintService(config)
    try:
        result = fabricate(part, overrides, service, load_printer_profile())
    except PhysLabError as exc:
        _echo_error(exc)
        raise typer.Exit(code=1) from exc
    finally:
        service.close()
    if not result.ok or result.job_id is None:
        typer.echo("\n".join(result.issues) or "print was not submitted", err=True)
        raise typer.Exit(code=1)
    typer.echo(result.job_id)


def _parse_params(params: str) -> dict[str, object]:
    loaded = json.loads(params)
    if not isinstance(loaded, dict):
        raise json.JSONDecodeError("expected an object", params, 0)
    return dict(loaded)


def _echo_error(exc: PhysLabError) -> None:
    tools = exc.details.get("tools")
    if isinstance(tools, list):
        for name in tools:
            typer.echo(str(name))
    typer.echo(exc.message, err=True)
