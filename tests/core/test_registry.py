"""Generic registry."""

from __future__ import annotations

import pytest

from physlab.core.errors import ConfigError
from physlab.core.registry import Registry


def _scale(value: int, *, factor: int = 2) -> int:
    return value * factor


def test_register_create_and_duplicate() -> None:
    registry: Registry[int] = Registry()
    registry.register("scale", _scale)
    assert registry.names() == ("scale",)
    assert registry.create("scale", value=3, factor=4) == 12
    with pytest.raises(ConfigError, match="scale"):
        registry.register("scale", _scale)
    with pytest.raises(ConfigError, match="missing"):
        registry.create("missing")
    with pytest.raises(ConfigError):
        registry.register("", _scale)
