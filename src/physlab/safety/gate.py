"""Presence check for real-world approval tokens.

T12 decides whether a token matches the plan or protocol that was approved.
This module only rejects a real-world call that arrives with no token.
"""

from __future__ import annotations

from pathlib import Path

from physlab.core.errors import ApprovalRequired, SafetyViolation
from physlab.core.types import Mode


def gate(action: str, *, mode: Mode, approval_token: str | None) -> None:
    """Require a non-empty approval token before a real-world action.

    Simulation returns immediately. The token is not checked against a file
    hash here. ``physlab approve`` is how a human creates one, and T12 is
    what will verify it.

    Args:
        action: Tool or command name stored on the error.
        mode: ``sim`` or ``real``.
        approval_token: Token printed by ``physlab approve``.

    Raises:
        ApprovalRequired: ``mode`` is real and the token is missing or blank.
    """
    if mode is not Mode.REAL:
        return
    if approval_token is None or not approval_token.strip():
        raise ApprovalRequired(
            "Real mode requires an approval token",
            details={"action": action},
        )


def assert_not_protected(path: Path, configs_dir: Path) -> None:
    """Reject a path that lives in the chemistry or safety config tree.

    Tools and ``physlab approve`` may read a plan the human names. They must
    not take a path inside those two directories, which are human-maintained.

    Args:
        path: Candidate file.
        configs_dir: Physlab configs directory.

    Raises:
        SafetyViolation: ``path`` is the chem directory, the safety directory,
            or a file inside either one.
    """
    resolved = path.resolve()
    for name in ("chem", "safety"):
        root = (configs_dir / name).resolve()
        if resolved == root or root in resolved.parents:
            raise SafetyViolation(
                "Protected config paths cannot be approved or written",
                details={"path": str(resolved)},
            )
