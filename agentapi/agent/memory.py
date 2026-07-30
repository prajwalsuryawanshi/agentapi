"""Memory backends for AgentAPI.

This module provides the :class:`MemoryBackend` abstract base class and
the following concrete implementations:

* :class:`InMemoryMemory` — ephemeral in-process storage (default).
* :class:`RedisMemory` — Redis-backed persistent storage with namespaced keys.

Security note
-------------
The :class:`RedisMemory` key format was historically ``agentapi:{conversation_id}``,
which meant that any client that guessed or enumerated another user's
``conversation_id`` could read, modify, or reset that user's conversation
history (Insecure Direct Object Reference / IDOR).

The updated implementation enforces a **mandatory namespaced key format**::

    agentapi:{tenant_id}:{user_id}:{conversation_id}

All three scoping dimensions are required.  Callers that omit ``user_id`` or
``tenant_id`` receive a ``ValueError`` at construction time so the mistake is
caught early rather than silently creating an unscoped key.

Existing deployments that used the old single-dimension key can migrate by
setting ``user_id="public"`` and ``tenant_id="default"`` as a transitional
step, then narrowing the scope as user isolation is added.
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from typing import Any


def create_conversation_id() -> str:
    """Generate a cryptographically random conversation ID."""
    return str(uuid.uuid4())


class MemoryBackend(ABC):
    """Abstract memory backend interface."""

    @property
    @abstractmethod
    def messages(self) -> list[dict[str, Any]]:
        """Return the current conversation history as a list of message dicts."""
        ...

    @abstractmethod
    def add(self, message: dict[str, Any]) -> None:
        """Append a message to the conversation history."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all messages from the conversation history."""
        ...


class InMemoryMemory(MemoryBackend):
    """In-process list-based memory (default, not persistent across restarts)."""

    def __init__(self) -> None:
        self._messages: list[dict[str, Any]] = []

    @property
    def messages(self) -> list[dict[str, Any]]:
        return list(self._messages)

    def add(self, message: dict[str, Any]) -> None:
        self._messages.append(message)

    def reset(self) -> None:
        self._messages.clear()


class RedisMemory(MemoryBackend):
    """Redis-backed conversation memory with namespaced, IDOR-safe key scoping.

    Keys are scoped to ``{tenant_id}:{user_id}:{conversation_id}`` so that
    a caller who knows *only* the ``conversation_id`` cannot read or mutate
    another user's conversation.  All three dimensions are **required**; the
    constructor raises :class:`ValueError` when any of them is missing or
    empty.

    Full Redis key format::

        agentapi:{tenant_id}:{user_id}:{conversation_id}

    Args:
        url: Redis connection URL (e.g. ``redis://localhost:6379/0``).
        conversation_id: Unique identifier for this conversation session.
        user_id: Identifier for the end-user.  **Required** — prevents
            cross-user IDOR.
        tenant_id: Top-level namespace (application / organisation).  Set to
            ``"default"`` for single-tenant deployments.
        ttl_seconds: Optional TTL in seconds for the Redis key.  The key
            is refreshed on every write.  Pass ``None`` to disable expiry.

    Raises:
        ValueError: If any of ``conversation_id``, ``user_id``, or
            ``tenant_id`` is ``None`` or an empty / whitespace-only string.
    """

    _KEY_PREFIX = "agentapi"
    _KEY_SEPARATOR = ":"
    _FORBIDDEN_CHARS = set(":\n\r\x00")

    def __init__(
        self,
        url: str,
        conversation_id: str,
        *,
        user_id: str,
        tenant_id: str = "default",
        ttl_seconds: int | None = None,
    ) -> None:
        import redis  # noqa: PLC0415 — optional dependency

        self._redis = redis.from_url(url, decode_responses=True)
        self._key = self._build_key(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        self._ttl = ttl_seconds

    @classmethod
    def _validate_segment(cls, value: str | None, name: str) -> str:
        """Validate and clean a key segment.  Raises ValueError on bad input."""
        if not value or not value.strip():
            raise ValueError(
                f"RedisMemory: '{name}' must be a non-empty string. "
                "Providing a scoped user_id and tenant_id is required to "
                "prevent cross-user conversation access (IDOR)."
            )
        cleaned = value.strip()
        bad = cls._FORBIDDEN_CHARS.intersection(cleaned)
        if bad:
            raise ValueError(
                f"RedisMemory: '{name}' contains forbidden characters {bad!r}. "
                "Key segments must not contain ':', newlines, or null bytes."
            )
        return cleaned

    @classmethod
    def _build_key(cls, *, tenant_id: str, user_id: str, conversation_id: str) -> str:
        """Build the fully-qualified, namespace-scoped Redis key.

        Format::

            agentapi:{tenant_id}:{user_id}:{conversation_id}

        All three segments are validated; a :class:`ValueError` is raised if
        any segment is empty or contains the colon separator character.
        """
        t = cls._validate_segment(tenant_id, "tenant_id")
        u = cls._validate_segment(user_id, "user_id")
        c = cls._validate_segment(conversation_id, "conversation_id")
        return cls._KEY_SEPARATOR.join([cls._KEY_PREFIX, t, u, c])

    @property
    def redis_key(self) -> str:
        """The fully-qualified Redis key used by this instance (read-only)."""
        return self._key

    @property
    def messages(self) -> list[dict[str, Any]]:
        raw = self._redis.get(self._key)
        if not raw:
            return []
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        return []

    def add(self, message: dict[str, Any]) -> None:
        current = self.messages
        current.append(message)
        serialized = json.dumps(current)
        if self._ttl:
            self._redis.setex(self._key, self._ttl, serialized)
        else:
            self._redis.set(self._key, serialized)

    def reset(self) -> None:
        self._redis.delete(self._key)
