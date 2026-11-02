"""Shared pytest hooks.

Tests marked ``assets`` need ``assets/mujoco_menagerie`` from
``physlab assets fetch``. They are skipped, with that command in the message,
when the checkout is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_SKIP_REASON = "assets not fetched; run `physlab assets fetch`"


def _menagerie_present() -> bool:
    root = Path(__file__).resolve().parents[1]
    return (root / "assets" / "mujoco_menagerie" / ".git").is_dir()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip asset tests when MuJoCo Menagerie has not been fetched.

    Args:
        config: Pytest config. Unused.
        items: Collected tests.
    """
    del config
    if _menagerie_present():
        return
    skip = pytest.mark.skip(reason=_SKIP_REASON)
    for item in items:
        if "assets" in item.keywords:
            item.add_marker(skip)
