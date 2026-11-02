"""JSONL event log."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from physlab.core.errors import PhysLabError
from physlab.core.events import Actor, EventLog, new_run_id
from physlab.core.types import Mode


def test_run_id_format_and_seed() -> None:
    moment = datetime(2026, 11, 2, 12, 30, 45, tzinfo=UTC)
    first = new_run_id(now=moment, seed=0)
    second = new_run_id(now=moment, seed=0)
    assert first == second
    assert first.startswith("20261102-123045-")
    assert len(first.split("-")[2]) == 6
    fresh = new_run_id(now=moment)
    assert fresh.startswith("20261102-123045-")
    with pytest.raises(PhysLabError):
        new_run_id(now=datetime(2026, 11, 2, 12, 30, 45))


def test_event_log_round_trip(tmp_path: Path) -> None:
    log = EventLog(tmp_path, mode=Mode.SIM, seed=1)
    assert log.read_all() == []
    written = log.record(
        Actor.AGENT,
        "move_pose",
        params={"target_frame": "world", "nested": {"ok": True}},
        result={"success": True},
    )
    again = log.record(Actor.SYSTEM, "note", mode=Mode.REAL)
    loaded = log.read_all()
    assert [event.action for event in loaded] == ["move_pose", "note"]
    assert loaded[0].params == {"target_frame": "world", "nested": {"ok": True}}
    assert loaded[0].result == {"success": True}
    assert loaded[0].actor is Actor.AGENT
    assert loaded[0].mode is Mode.SIM
    assert loaded[0].run_id == log.run_id
    assert loaded[0].timestamp.tzinfo is not None
    assert loaded[1].mode is Mode.REAL
    assert loaded[1].params == {}
    assert loaded[1].result is None
    assert again.run_id == written.run_id
    text = log.path.read_text(encoding="utf-8")
    assert text.count("\n") == 2


def test_event_log_rejects_bad_records(tmp_path: Path) -> None:
    log = EventLog(tmp_path, run_id="20261102-123045-abc123")
    with pytest.raises(PhysLabError):
        log.record(Actor.HUMAN, "")
    with pytest.raises(PhysLabError):
        log.record(Actor.HUMAN, "look", params={"path": Path("/tmp")})
    with pytest.raises(PhysLabError):
        EventLog(tmp_path, run_id="not-an-id")
    log.path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(PhysLabError):
        log.read_all()
