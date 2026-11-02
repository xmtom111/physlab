"""Clone MuJoCo Menagerie at the commit pinned in ``configs/assets.lock``.

``menagerie_commit: latest`` clones the ``main`` branch and writes the resolved
commit back into the lock file. A later call that is already at that commit
does not touch the network.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import yaml

from physlab.core.config import load_config
from physlab.core.errors import AssetMissing, ConfigError
from physlab.core.logging import get_logger

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie"
"""Upstream repository cloned into ``assets/mujoco_menagerie``."""

LATEST = "latest"
_LOCK_NAME = "assets.lock"
_DEST_NAME = "mujoco_menagerie"
_logger = get_logger(__name__)


def fetch_assets(
    configs_dir: Path | None = None,
    assets_dir: Path | None = None,
    repo_url: str = MENAGERIE_URL,
) -> str:
    """Clone or update MuJoCo Menagerie and return the checked-out commit.

    Args:
        configs_dir: Directory that contains ``assets.lock``. The physlab
            config is used when omitted.
        assets_dir: Directory that will contain ``mujoco_menagerie``.
        repo_url: Git URL. The default is the public menagerie repository.
            Tests pass a local repository.

    Returns:
        Full commit hash now checked out.

    Raises:
        AssetMissing: ``git`` is missing, a git command fails, or the
            destination exists and is not a git checkout.
        ConfigError: ``assets.lock`` does not contain ``menagerie_commit``.
    """
    git = shutil.which("git")
    if git is None:
        raise AssetMissing("git is not installed; cannot clone mujoco_menagerie")
    config_dir, asset_dir = _dirs(configs_dir, assets_dir)
    lock_path = config_dir / _LOCK_NAME
    requested = _read_commit(lock_path)
    destination = asset_dir / _DEST_NAME
    if requested == LATEST:
        _ensure_clone(git, repo_url, destination)
        _run(git, destination, ["fetch", "--depth", "1", "origin", "main"])
        _run(git, destination, ["checkout", "--detach", "FETCH_HEAD"])
        resolved = _rev_parse(git, destination)
        _write_commit(lock_path, resolved)
        _logger.info("menagerie_resolved", commit=resolved, path=str(destination))
        return resolved
    if _is_checked_out(git, destination, requested):
        _logger.info("menagerie_unchanged", commit=requested, path=str(destination))
        return requested
    _ensure_clone(git, repo_url, destination)
    _run(git, destination, ["fetch", "--depth", "1", "origin", requested])
    _run(git, destination, ["checkout", "--detach", requested])
    resolved = _rev_parse(git, destination)
    if resolved != requested:
        raise AssetMissing(
            "Checked-out menagerie commit does not match assets.lock",
            details={"expected": requested, "actual": resolved},
        )
    _logger.info("menagerie_checked_out", commit=resolved, path=str(destination))
    return resolved


def _dirs(configs_dir: Path | None, assets_dir: Path | None) -> tuple[Path, Path]:
    if configs_dir is not None and assets_dir is not None:
        return configs_dir, assets_dir
    loaded = load_config()
    return configs_dir or loaded.configs_dir, assets_dir or loaded.assets_dir


def _read_commit(path: Path) -> str:
    if not path.is_file():
        raise ConfigError("assets.lock was not found", details={"path": str(path)})
    loaded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or "menagerie_commit" not in loaded:
        raise ConfigError(
            "assets.lock must set menagerie_commit",
            details={"path": str(path)},
        )
    commit = loaded["menagerie_commit"]
    if not isinstance(commit, str) or not commit.strip():
        raise ConfigError(
            "menagerie_commit must be a non-empty string",
            details={"path": str(path)},
        )
    return commit.strip()


def _write_commit(path: Path, commit: str) -> None:
    text = path.read_text(encoding="utf-8")
    replaced, count = re.subn(
        r"(?m)^menagerie_commit:\s*.*$",
        f"menagerie_commit: {commit}",
        text,
        count=1,
    )
    if count != 1:
        raise ConfigError(
            "Could not update menagerie_commit in assets.lock",
            details={"path": str(path)},
        )
    path.write_text(replaced, encoding="utf-8")


def _ensure_clone(git: str, repo_url: str, destination: Path) -> None:
    if destination.exists():
        if not (destination / ".git").exists():
            raise AssetMissing(
                "Asset directory exists and is not a git checkout",
                details={"path": str(destination)},
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(git, None, ["clone", "--origin", "origin", repo_url, str(destination)])


def _is_checked_out(git: str, destination: Path, commit: str) -> bool:
    if not (destination / ".git").exists():
        return False
    return _rev_parse(git, destination) == commit


def _rev_parse(git: str, destination: Path) -> str:
    return _run(git, destination, ["rev-parse", "HEAD"])


def _run(git: str, cwd: Path | None, args: list[str]) -> str:
    completed = subprocess.run(
        [git, *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssetMissing(
            "git command failed",
            details={"args": [git, *args], "stderr": completed.stderr.strip()},
        )
    return completed.stdout.strip()
