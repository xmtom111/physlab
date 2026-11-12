"""structlog configuration for physlab.

Call :func:`configure_logging` once at process start. :func:`get_logger`
returns a logger bound with the module name. Library code must log through
that logger instead of ``print``.
"""

from __future__ import annotations

import logging as stdlib_logging
import sys
from typing import TextIO, cast

import structlog
from structlog.typing import FilteringBoundLogger

from physlab.core.errors import ConfigError


def configure_logging(
    level: str | int,
    json: bool = False,
    *,
    stream: TextIO | None = None,
) -> None:
    """Configure process-wide structlog output.

    JSON lines are written when ``json`` is true. Otherwise a console renderer
    without color is used. This replaces any previous structlog configuration
    for the process. structlog keeps its configuration in process globals; this
    function is the only place physlab writes that state.

    Args:
        level: Log level name such as ``"INFO"``, or a stdlib numeric level.
        json: Emit one JSON object per line when true.
        stream: Where lines are written. Standard output is used when omitted.
            The MCP stdio server passes standard error so logs are not framed
            as protocol messages.

    Raises:
        ConfigError: ``level`` is not a known logging level.
    """
    numeric_level = _numeric_level(level)
    processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.format_exc_info,
    ]
    if json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=False))
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout if stream is None else stream),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> FilteringBoundLogger:
    """Return a structlog logger tagged with ``name``.

    Args:
        name: Logger name, usually ``__name__``.

    Returns:
        Bound logger. Events include a ``logger`` field set to ``name``.
    """
    # PrintLoggerFactory does not take a name. Bind it into the event context.
    bound = structlog.get_logger().bind(logger=name)
    return cast(FilteringBoundLogger, bound)


def _numeric_level(level: str | int) -> int:
    if isinstance(level, int):
        if level < 0:
            raise ConfigError("Log level must be non-negative", details={"level": level})
        return level
    mapping = stdlib_logging.getLevelNamesMapping()
    numeric = mapping.get(level.upper())
    if numeric is None:
        raise ConfigError("Unknown log level", details={"level": level})
    return int(numeric)
