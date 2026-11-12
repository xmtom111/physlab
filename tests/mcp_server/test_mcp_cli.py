"""Command line coverage for the MCP server and the approval token."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from physlab.cli import app


def test_serve_help_names_both_transports() -> None:
    result = CliRunner().invoke(app, ["mcp", "serve", "--help"])
    assert result.exit_code == 0
    assert "stdio" in result.stdout
    assert "streamable-http" in result.stdout
    assert "--config" in result.stdout


def test_unknown_transport_exits() -> None:
    result = CliRunner().invoke(app, ["mcp", "serve", "--transport", "sse"])
    assert result.exit_code == 2


def test_approve_prints_a_token_and_refuses_protected_configs(tmp_path: Path) -> None:
    plan = tmp_path / "plan.yaml"
    plan.write_text("id: demo\n", encoding="utf-8")
    runs = tmp_path / "runs"
    minted = CliRunner().invoke(app, ["approve", str(plan)], env={"PHYSLAB_RUN_DIR": str(runs)})
    assert minted.exit_code == 0
    token = minted.stdout.strip()
    assert len(token) == 32
    record = json.loads((runs / "approvals" / f"{token}.json").read_text(encoding="utf-8"))
    assert record["token"] == token
    assert record["sha256"]
    refused = CliRunner().invoke(app, ["approve", "configs/chem/reagents.yaml"])
    assert refused.exit_code == 1
    assert "Protected" in refused.stderr
