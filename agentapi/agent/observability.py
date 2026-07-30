"""Structured observability helpers for AgentAPI."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any


AgentEvent = dict[str, Any]
AgentEventHook = Callable[[AgentEvent], Any]
logger = logging.getLogger(__name__)


def build_event(event: str, **fields: Any) -> AgentEvent:
    """Create a structured event with a stable timestamp field."""

    return {
        "event": event,
        "timestamp": datetime.now(UTC).isoformat(),
        **fields,
    }


async def emit_event(hooks: list[AgentEventHook], event: AgentEvent) -> None:
    """Send an event to all registered hooks without mutating the payload."""

    for hook in tuple(hooks):
        try:
            result = hook(dict(event))
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.warning(
                "AgentAPI event hook failed for %s",
                event.get("event", "<unknown_event>"),
                exc_info=True,
            )
