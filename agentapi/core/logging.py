"""Structured logging utilities for AgentAPI.

Provides a ``configure_logging`` helper that sets up consistent,
structured log output (key=value style) across all agentapi modules.
Callers can also retrieve named loggers via ``get_logger``.

Usage::

    from agentapi.core.logging import configure_logging
    configure_logging(level="DEBUG")
"""

from __future__ import annotations

import logging
import sys
from typing import Literal

__all__ = ["configure_logging", "get_logger"]

LevelName = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"


class _StructuredFormatter(logging.Formatter):
    """Formatter that emits key=value pairs after the base message."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        base = super().format(record)
        extras: list[str] = []
        skip = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "asctime",
        }
        for key, value in record.__dict__.items():
            if key not in skip and not key.startswith("_"):
                extras.append(f"{key}={value!r}")
        if extras:
            return f"{base} | {' '.join(extras)}"
        return base


def configure_logging(
    level: LevelName | str = "INFO",
    *,
    stream: object = None,
    structured: bool = True,
) -> None:
    """Configure the root ``agentapi`` logger.

    Call this once at application startup, before creating any
    :class:`~agentapi.core.app.AgentAPI` or :class:`~agentapi.agent.agent.Agent`
    instances, to ensure all lifecycle events are captured.

    Args:
        level: Logging level name (e.g. ``"DEBUG"``, ``"INFO"``).
            Defaults to ``"INFO"``.
        stream: Output stream for the handler.  Defaults to ``sys.stdout``.
        structured: If ``True`` (the default), emit ``key=value`` pairs for
            any extra fields attached to log records.  Set to ``False`` for
            plain human-readable output.
    """
    if stream is None:
        stream = sys.stdout

    root_logger = logging.getLogger("agentapi")
    if root_logger.handlers:
        # Avoid adding duplicate handlers if called multiple times.
        root_logger.handlers.clear()

    handler = logging.StreamHandler(stream)  # type: ignore[arg-type]
    if structured:
        formatter: logging.Formatter = _StructuredFormatter(_FORMAT, datefmt=_DATEFMT)
    else:
        formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``agentapi`` namespace.

    Args:
        name: Dot-separated sub-name appended to ``"agentapi."``.  Pass
            ``__name__`` from the calling module for idiomatic usage.

    Returns:
        A :class:`logging.Logger` instance.
    """
    return logging.getLogger(f"agentapi.{name}" if not name.startswith("agentapi") else name)
