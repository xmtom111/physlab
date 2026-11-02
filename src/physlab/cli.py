"""Command line interface for physlab."""

from __future__ import annotations

from typing import Annotated

import typer
import yaml

from physlab import __version__
from physlab.core.config import load_config
from physlab.robots.cli import register

app = typer.Typer(add_completion=False, no_args_is_help=True)
config_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(config_app, name="config")


def _version_callback(value: bool) -> None:
    """Print the version and exit when ``--version`` is set."""
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Print the physlab version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Operate robots, fabrication, and experiments from the command line."""


@config_app.callback()
def config_main() -> None:
    """Inspect the effective physlab configuration."""


@config_app.command("show")
def config_show() -> None:
    """Print the effective configuration as YAML."""
    loaded = load_config()
    rendered = yaml.safe_dump(loaded.model_dump(mode="json"), sort_keys=False).rstrip()
    typer.echo(rendered)


register(app)
