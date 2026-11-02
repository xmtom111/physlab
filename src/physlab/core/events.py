"""Append-only JSONL event log for one run.

Each run lives at ``run_dir/<run_id>/events.jsonl``. Run ids look like
``YYYYMMDD-HHMMSS-<6 hex>``. Pass ``seed`` when the hex suffix must be
reproducible; otherwise the suffix comes from the operating system's entropy
so two runs started in the same second do not collide.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from physlab.core.errors import PhysLabError
from physlab.core.types import Mode

_RUN_ID_WIDTH = 6


class Actor(StrEnum):
    """Who performed the recorded action."""

    AGENT = "agent"
    HUMAN = "human"
    SYSTEM = "system"


class Event(BaseModel):
    """One recorded action.

    Attributes:
        timestamp: Timezone-aware UTC timestamp.
        run_id: Run that owns this event.
        actor: Agent, human, or system.
        action: Short verb, for example ``move_pose``.
        params: JSON object of inputs.
        result: JSON object of outputs, or ``None`` when the action has none.
        mode: Simulation or real hardware.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime
    run_id: str
    actor: Actor
    action: str
    params: dict[str, Any]
    result: dict[str, Any] | None
    mode: Mode

    @field_validator("timestamp")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise PhysLabError("Event timestamp must be timezone-aware UTC")
        return value.astimezone(UTC)

    @field_validator("run_id")
    @classmethod
    def _run_id(cls, value: str) -> str:
        return _check_run_id(value)

    @field_validator("action")
    @classmethod
    def _action(cls, value: str) -> str:
        if not value:
            raise PhysLabError("Event action must be a non-empty string")
        return value


class EventLog:
    """JSONL log for a single run."""

    def __init__(
        self,
        run_dir: Path,
        run_id: str | None = None,
        mode: Mode = Mode.SIM,
        seed: int | None = None,
    ) -> None:
        """Create the run directory and remember where events are written.

        Args:
            run_dir: Parent directory of all runs.
            run_id: Existing id to reuse. A new id is generated when omitted.
            mode: Default mode stored on events that do not override it.
            seed: Seed for the hex suffix. Ignored when ``run_id`` is set.
                When both are omitted, the suffix is drawn from OS entropy.

        Raises:
            PhysLabError: ``run_id`` does not match ``YYYYMMDD-HHMMSS-<6 hex>``.
        """
        self.run_dir = run_dir
        self.run_id = _check_run_id(run_id) if run_id is not None else new_run_id(seed=seed)
        self.mode = mode
        self.path = run_dir / self.run_id / "events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        actor: Actor,
        action: str,
        params: Mapping[str, object] | None = None,
        result: Mapping[str, object] | None = None,
        mode: Mode | None = None,
    ) -> Event:
        """Append one event and return it.

        Args:
            actor: Who performed the action.
            action: Short non-empty verb.
            params: JSON-serializable inputs. ``None`` stores an empty object.
            result: JSON-serializable outputs. ``None`` stores no result.
            mode: Override the log's default mode for this event.

        Returns:
            The event that was written.

        Raises:
            PhysLabError: ``action`` is empty, or ``params`` / ``result`` is not
                a JSON object.
        """
        event = Event(
            timestamp=datetime.now(UTC),
            run_id=self.run_id,
            actor=actor,
            action=action,
            params=_json_object({} if params is None else params, "params"),
            result=None if result is None else _json_object(result, "result"),
            mode=self.mode if mode is None else mode,
        )
        line = json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return event

    def read_all(self) -> list[Event]:
        """Read every event in write order.

        Returns:
            Events parsed from the JSONL file. Empty when nothing was recorded.

        Raises:
            PhysLabError: A line is not valid JSON or does not match
                :class:`Event`.
        """
        if not self.path.is_file():
            return []
        events: list[Event] = []
        text = self.path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            events.append(_parse_line(self.path, line_number, line))
        return events


def new_run_id(now: datetime | None = None, seed: int | None = None) -> str:
    """Build a run id ``YYYYMMDD-HHMMSS-<6 hex>``.

    Args:
        now: UTC timestamp. The current time is used when omitted.
        seed: When set, the hex suffix comes from a numpy Generator seeded
            with this value. When omitted, the suffix comes from
            ``secrets.token_hex``.

    Returns:
        Run id using a UTC timestamp.

    Raises:
        PhysLabError: ``now`` has no timezone.
    """
    moment = datetime.now(UTC) if now is None else now
    if moment.tzinfo is None:
        raise PhysLabError("Run id timestamp must be timezone-aware")
    stamp = moment.astimezone(UTC).strftime("%Y%m%d-%H%M%S")
    suffix = _hex_suffix(seed)
    return f"{stamp}-{suffix}"


def _hex_suffix(seed: int | None) -> str:
    if seed is None:
        return secrets.token_hex(3)
    generator = np.random.default_rng(seed)
    digits = generator.integers(0, 16, size=_RUN_ID_WIDTH)
    return "".join(f"{int(digit):x}" for digit in digits)


def _check_run_id(run_id: str) -> str:
    parts = run_id.split("-")
    if len(parts) != 3 or len(parts[0]) != 8 or len(parts[1]) != 6 or len(parts[2]) != 6:
        raise PhysLabError(
            "Run id must look like YYYYMMDD-HHMMSS-<6 hex>",
            details={"run_id": run_id},
        )
    date_part, time_part, suffix = parts
    if not date_part.isdigit() or not time_part.isdigit():
        raise PhysLabError(
            "Run id must look like YYYYMMDD-HHMMSS-<6 hex>",
            details={"run_id": run_id},
        )
    if any(character not in "0123456789abcdef" for character in suffix):
        raise PhysLabError(
            "Run id suffix must be 6 lowercase hex characters",
            details={"run_id": run_id},
        )
    return run_id


def _json_object(value: Mapping[str, object], label: str) -> dict[str, Any]:
    # Any: json.loads returns open JSON values (dicts, lists, scalars).
    try:
        encoded = json.dumps(dict(value))
    except TypeError as exc:
        raise PhysLabError(
            f"{label} must be JSON serializable",
            details={"reason": str(exc)},
        ) from exc
    decoded: object = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise PhysLabError(f"{label} must be a JSON object")
    cleaned: dict[str, Any] = {}
    for key, item in decoded.items():
        if not isinstance(key, str):
            raise PhysLabError(f"{label} keys must be strings")
        cleaned[key] = item
    return cleaned


def _parse_line(path: Path, line_number: int, line: str) -> Event:
    try:
        payload: object = json.loads(line)
        return Event.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, PhysLabError) as exc:
        raise PhysLabError(
            "Event log line is invalid",
            details={"path": str(path), "line": line_number, "reason": str(exc)},
        ) from exc
