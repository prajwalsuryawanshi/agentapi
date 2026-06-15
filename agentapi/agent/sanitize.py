"""Conversation history sanitization utilities for AgentAPI.

This module provides tools for detecting and removing *orphaned tool calls*
from a conversation history before it is sent to an LLM provider.

An orphaned tool call occurs when a ``tool_calls`` entry in an assistant
message has no corresponding ``role: tool`` message with a matching
``tool_call_id``.  This commonly happens when a user disconnects mid-stream
or when a partial history is loaded from a persistent store after a crash.

Providers such as OpenAI and Anthropic treat incomplete tool-call sequences
as a protocol violation and raise API errors.  This module detects and
drops those incomplete sequences so the conversation can continue cleanly.

Usage::

    from agentapi.agent.sanitize import sanitize_conversation_history

    clean = sanitize_conversation_history(raw_history)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def sanitize_conversation_history(
    messages: list[dict[str, Any]],
    *,
    ignore_incomplete: bool = True,
) -> list[dict[str, Any]]:
    """Remove orphaned tool-call blocks from a conversation history.

    An *orphaned* tool call is an entry inside an assistant message's
    ``tool_calls`` list for which no ``role: "tool"`` reply with a matching
    ``tool_call_id`` exists anywhere later in the history.

    The function also removes the entire assistant ``tool_calls`` turn when
    all of its individual tool calls are orphaned, since an assistant message
    with an empty ``tool_calls`` list is itself a protocol violation for most
    providers.

    Args:
        messages: A list of conversation message dicts as returned by
            :attr:`~agentapi.agent.memory.MemoryBackend.messages` or built
            inline.  The list is **not** modified in place.
        ignore_incomplete: If ``True`` (the default) silently drop orphaned
            blocks and emit ``DEBUG`` log lines.  If ``False`` raise a
            :class:`ValueError` listing the orphaned call IDs instead.

    Returns:
        A new list of messages with all orphaned tool-call references removed.

    Raises:
        ValueError: When ``ignore_incomplete=False`` and orphaned calls are
            detected.
    """
    # Collect all tool_call_ids that have a corresponding tool-result message.
    result_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") == "tool":
            call_id = msg.get("tool_call_id")
            if call_id:
                result_ids.add(call_id)

    orphaned_ids: list[str] = []
    sanitized: list[dict[str, Any]] = []

    for msg in messages:
        if msg.get("role") != "assistant" or "tool_calls" not in msg:
            sanitized.append(msg)
            continue

        # Filter out individual tool_call entries that have no matching result.
        clean_calls = [
            call for call in msg["tool_calls"]
            if _call_id(call) in result_ids
        ]
        dropped = [
            _call_id(call) for call in msg["tool_calls"]
            if _call_id(call) not in result_ids
        ]
        orphaned_ids.extend(dropped)

        if not clean_calls:
            # All calls in this turn are orphaned — drop the entire assistant
            # turn so the history remains coherent.
            logger.debug(
                "[AgentAPI] Dropping orphaned assistant tool_calls turn",
                extra={"dropped_ids": dropped},
            )
            continue

        if dropped:
            logger.debug(
                "[AgentAPI] Dropping orphaned tool_call entries from assistant turn",
                extra={"dropped_ids": dropped, "kept_ids": [_call_id(c) for c in clean_calls]},
            )

        # Rebuild the assistant message with only the clean calls.
        clean_msg = {**msg, "tool_calls": clean_calls}
        sanitized.append(clean_msg)

    if orphaned_ids and not ignore_incomplete:
        raise ValueError(
            f"Orphaned tool call IDs detected in conversation history: {orphaned_ids}. "
            "Ensure every tool_call has a corresponding tool-result message, or set "
            "ignore_incomplete=True to drop them silently."
        )

    return sanitized


def _call_id(call: dict[str, Any]) -> str:
    """Extract the tool call ID from a tool_calls entry (empty string as fallback)."""
    return call.get("id", "")
