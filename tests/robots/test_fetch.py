"""Asset lock checkout using a local git repository."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from physlab.core.errors import AssetMissing, ConfigError
from physlab.robots.fetch import fetch_assets

_GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "physlab",
    "GIT_AUTHOR_EMAIL": "physlab@example.com",
    "GIT_COMMITTER_NAME": "physlab",
    "GIT_COMMITTER_EMAIL": "physlab@example.com",
}


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env=_GIT_ENV,
    )
    return completed.stdout.strip()


def _remote(path: Path) -> tuple[str, str]:
    path.mkdir()
    _git(path, "init", "-b", "main")
    (path / "README").write_text("one\n", encoding="utf-8")
    _git(path, "add", "README")
    _git(path, "commit", "-m", "one")
    first = _git(path, "rev-parse", "HEAD")
    (path / "README").write_text("two\n", encoding="utf-8")
    _git(path, "add", "README")
    _git(path, "commit", "-m", "two")
    second = _git(path, "rev-parse", "HEAD")
    return first, second


def _lock(configs: Path, commit: str) -> Path:
    configs.mkdir()
    path = configs / "assets.lock"
    path.write_text(
        f"# Pin for third-party assets.\nmenagerie_commit: {commit}\n",
        encoding="utf-8",
    )
    return path


def test_latest_is_written_back_and_repeat_is_idle(tmp_path: Path) -> None:
    first, second = _remote(tmp_path / "remote")
    configs = tmp_path / "configs"
    assets = tmp_path / "assets"
    lock = _lock(configs, "latest")
    resolved = fetch_assets(configs, assets, repo_url=str(tmp_path / "remote"))
    assert resolved == second
    assert lock.read_text(encoding="utf-8").endswith(f"menagerie_commit: {second}\n")
    assert _git(assets / "mujoco_menagerie", "rev-parse", "HEAD") == second
    text = lock.read_text(encoding="utf-8")
    again = fetch_assets(configs, assets, repo_url=str(tmp_path / "remote"))
    assert again == second
    assert lock.read_text(encoding="utf-8") == text
    assert first != second


def test_pinned_commit_is_checked_out(tmp_path: Path) -> None:
    first, second = _remote(tmp_path / "remote")
    configs = tmp_path / "configs"
    assets = tmp_path / "assets"
    _lock(configs, "latest")
    fetch_assets(configs, assets, repo_url=str(tmp_path / "remote"))
    lock = configs / "assets.lock"
    lock.write_text(f"menagerie_commit: {first}\n", encoding="utf-8")
    checked = fetch_assets(configs, assets, repo_url=str(tmp_path / "remote"))
    assert checked == first
    assert _git(assets / "mujoco_menagerie", "rev-parse", "HEAD") == first
    assert fetch_assets(configs, assets, repo_url=str(tmp_path / "remote")) == first
    assert second


def test_missing_git_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("physlab.robots.fetch.shutil.which", lambda _name: None)
    with pytest.raises(AssetMissing, match="git is not installed"):
        fetch_assets(tmp_path, tmp_path)


def test_lock_must_name_a_commit(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "assets.lock").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="menagerie_commit"):
        fetch_assets(configs, tmp_path / "assets", repo_url="unused")
