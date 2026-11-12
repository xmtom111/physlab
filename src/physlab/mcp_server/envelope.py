"""JSON envelopes returned by every MCP tool."""

from __future__ import annotations

import base64
from enum import Enum
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from physlab.core.errors import PhysLabError


def success(data: object) -> dict[str, object]:
    """Wrap a successful payload.

    Args:
        data: JSON-ready value stored under ``data``.

    Returns:
        Envelope with ``ok`` true and ``error`` null.
    """
    return {"ok": True, "data": jsonable(data), "error": None}


def failure(exc: BaseException) -> dict[str, object]:
    """Wrap a domain error.

    Args:
        exc: Exception raised by a tool. ``PhysLabError.details`` is copied
            when it is present.

    Returns:
        Envelope with ``ok`` false and a typed ``error`` object.
    """
    details: dict[str, object] = {}
    raw = getattr(exc, "details", None)
    if isinstance(raw, dict):
        details = cast(dict[str, object], jsonable(raw))
    message = exc.message if isinstance(exc, PhysLabError) else str(exc)
    return {
        "ok": False,
        "data": None,
        "error": {"type": type(exc).__name__, "message": message, "details": details},
    }


def jsonable(value: object) -> object:
    """Convert ``value`` into objects ``json.dumps`` can encode.

    Args:
        value: Model, path, enum, container, or scalar.

    Returns:
        JSON scalars, lists, and string-keyed dicts.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseModel):
        return jsonable(value.model_dump(mode="json"))
    if isinstance(value, Enum):
        return jsonable(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return base64.standard_b64encode(value).decode("ascii")
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return str(value)
    return number


def for_log(value: object) -> object:
    """Drop image payloads before they are written to the event log.

    Args:
        value: Envelope or nested JSON value.

    Returns:
        The same structure with base64 image fields replaced by a byte count.
    """
    if isinstance(value, dict):
        trimmed: dict[str, object] = {}
        for key, item in value.items():
            if key in {"preview_png_base64", "plot_images"} or key.endswith("_base64"):
                trimmed[key] = {"omitted": True}
                continue
            trimmed[str(key)] = for_log(item)
        return trimmed
    if isinstance(value, list):
        return [for_log(item) for item in value]
    return value
