"""``physlab assets`` and ``physlab robots`` commands."""

from __future__ import annotations

from typing import Annotated

import typer

from physlab.core.config import load_config
from physlab.core.errors import AssetMissing, ConfigError, PhysLabError
from physlab.robots.catalog import RobotCatalog
from physlab.robots.fetch import fetch_assets
from physlab.robots.models import RobotSpec

assets_app = typer.Typer(add_completion=False, no_args_is_help=True)
robots_app = typer.Typer(add_completion=False, no_args_is_help=True)


def register(app: typer.Typer) -> None:
    """Attach the asset and robot command groups.

    Args:
        app: Root physlab Typer application.
    """
    app.add_typer(assets_app, name="assets")
    app.add_typer(robots_app, name="robots")


@assets_app.callback()
def assets_main() -> None:
    """Download third-party robot models."""


@assets_app.command("fetch")
def assets_fetch() -> None:
    """Clone MuJoCo Menagerie at the ``assets.lock`` commit."""
    try:
        commit = fetch_assets()
    except PhysLabError as exc:
        typer.echo(exc.message, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(commit)


@robots_app.callback()
def robots_main() -> None:
    """List and validate catalog robots."""


@robots_app.command("list")
def robots_list() -> None:
    """Print id, kind, availability, motion dof, and gripper."""
    catalog = _catalog()
    rows = [_row(spec) for spec in catalog.list()]
    header = ("id", "kind", "available", "dof", "gripper")
    widths = [len(cell) for cell in header]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    typer.echo(_format_row(header, widths))
    for row in rows:
        typer.echo(_format_row(row, widths))


@robots_app.command("check")
def robots_check(
    robot_id: Annotated[str | None, typer.Argument(help="Robot id to check.")] = None,
    all_robots: Annotated[
        bool,
        typer.Option("--all", help="Check every available robot."),
    ] = False,
) -> None:
    """Load each robot and confirm its joints, actuators, sites, and bodies."""
    if all_robots == (robot_id is not None):
        typer.echo("Pass a robot id or --all.", err=True)
        raise typer.Exit(code=2)
    catalog = _catalog()
    if all_robots:
        targets = [spec.id for spec in catalog.list() if spec.available]
    else:
        assert robot_id is not None
        targets = [robot_id]
    failed = False
    for target in targets:
        try:
            result = catalog.check(target)
        except ConfigError as exc:
            typer.echo(f"FAIL {target} {exc.message}")
            failed = True
            continue
        status = "OK" if result.ok else "FAIL"
        typer.echo(f"{status} {result.robot_id}")
        for message in result.messages:
            typer.echo(f"  {message}")
        failed = failed or not result.ok
    if failed:
        raise typer.Exit(code=1)


def _catalog() -> RobotCatalog:
    try:
        loaded = load_config()
        return RobotCatalog(loaded.configs_dir, loaded.assets_dir)
    except (ConfigError, AssetMissing) as exc:
        typer.echo(exc.message, err=True)
        raise typer.Exit(code=1) from exc


def _row(spec: RobotSpec) -> tuple[str, str, str, str, str]:
    """Return one table row for ``spec``."""
    gripper = spec.sim.gripper.actuator if spec.sim.gripper is not None else "-"
    dof = len(spec.sim.arm_joints) + len(spec.sim.base_joints)
    available = "yes" if spec.available else "no"
    return (spec.id, spec.kind.value, available, str(dof), gripper)


def _format_row(cells: tuple[str, ...], widths: list[int]) -> str:
    return "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(cells))
