"""structlog configuration."""

from __future__ import annotations

import json

import pytest

from physlab.core.errors import ConfigError
from physlab.core.logging import configure_logging, get_logger


def test_json_lines_and_level_filter(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", True)
    logger = get_logger("physlab.tests.logging")
    logger.debug("hidden", token=1)
    logger.info("visible", token=2)
    captured = capsys.readouterr().out.strip().splitlines()
    assert captured
    payload = json.loads(captured[-1])
    assert payload["event"] == "visible"
    assert payload["logger"] == "physlab.tests.logging"
    assert payload["token"] == 2
    assert payload["level"] == "info"
    assert "hidden" not in "\n".join(captured)


def test_console_renderer_and_bad_level(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(20, False)
    get_logger("physlab.tests.console").warning("heads_up")
    assert "heads_up" in capsys.readouterr().out
    with pytest.raises(ConfigError):
        configure_logging("loud")
    with pytest.raises(ConfigError):
        configure_logging(-1)
