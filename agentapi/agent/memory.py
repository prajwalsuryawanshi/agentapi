"""Conversation memory backends for agents."""

from __future__ import annotations

import json
from importlib import import_module
from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID, uuid4


def create_conversation_id() -> str:
    """Create a canonical UUIDv4 conversation ID."""

    return str(uuid4())


class MemoryBackend(ABC):
    """Abstract memory backend contract."""

    @property
    @abstractmethod
    def messages(self) -> list[dict[str, Any]]:
        """Return the current conversation messages."""

    @abstractmethod
    def add(self, message: dict[str, Any]) -> None:
        """Append one message to the conversation."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all stored messages for the conversation."""

    def for_conversation(self, conversation_id: str) -> "MemoryBackend":
        """Return a backend bound to a specific conversation.

        Backends that support multi-conversation resolution can override this
        to return a sibling/backend view for the given conversation. The
        default implementation fails fast so callers cannot silently assume
        conversation-scoped isolation when a custom backend does not support it.
        """

        raise NotImplementedError(
            f"{self.__class__.__name__} does not support conversation-scoped memory resolution"
        )


class InMemoryMemory(MemoryBackend):
    """Stores chat messages in process memory with per-conversation isolation.

    Supports multiple conversations keyed by UUID. Ideal for development and
    testing multi-user scenarios without external dependencies.
    """

    def __init__(
        self,
        conversation_id: str | None = None,
        *,
        _conversations: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        # Validate and normalize to canonical UUID string if provided; auto-generate otherwise.
        if conversation_id is not None:
            self.conversation_id = str(UUID(conversation_id))
        else:
            self.conversation_id = create_conversation_id()

        # Per-conversation message storage shared by sibling views when needed.
        self._conversations: dict[str, list[dict[str, Any]]] = _conversations or {}

        # Initialize this conversation only once so sibling views share history.
        if self.conversation_id not in self._conversations:
            self._conversations[self.conversation_id] = []

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self._conversations.get(self.conversation_id, [])

    def add(self, message: dict[str, Any]) -> None:
        if self.conversation_id not in self._conversations:
            self._conversations[self.conversation_id] = []
        self._conversations[self.conversation_id].append(message)

    def reset(self) -> None:
        self._conversations[self.conversation_id] = []

    def for_conversation(self, conversation_id: str) -> MemoryBackend:
        return InMemoryMemory(
            conversation_id=conversation_id,
            _conversations=self._conversations,
        )


class RedisMemory(MemoryBackend):
    """Redis-backed memory for multi-user and multi-worker deployments.

    Requires: `pip install redis`
    """

    def __init__(
        self,
        *,
        redis_url: str,
        conversation_id: str,
        user_id: str | None = None,
        tenant_id: str | None = None,
        ttl_seconds: int = 7 * 24 * 60 * 60,
        _redis_client: Any | None = None,
    ) -> None:
        # Validate and normalize to canonical UUID string.
        self._redis_url = redis_url
        self.conversation_id = str(UUID(conversation_id))
        self.user_id = user_id
        self.tenant_id = tenant_id
        self._ttl_seconds = ttl_seconds

        if _redis_client is not None:
            self._redis = _redis_client
            self._owns_redis_client = False
        else:
            try:
                redis_module = import_module("redis")
            except ImportError as exc:  # pragma: no cover - depends on optional dependency
                raise ImportError("redis package is required for RedisMemory. Install with: pip install redis") from exc

            self._redis = redis_module.Redis.from_url(redis_url, decode_responses=True)
            self._owns_redis_client = True

    @property
    def _messages_key(self) -> str:
        return f"conv:{self.conversation_id}:messages"

    @property
    def _meta_key(self) -> str:
        return f"conv:{self.conversation_id}:meta"

    def _ensure_meta(self) -> None:
        if self._redis.exists(self._meta_key):
            return

        mapping: dict[str, str] = {"conversation_id": self.conversation_id}
        if self.user_id is not None:
            mapping["user_id"] = self.user_id
        if self.tenant_id is not None:
            mapping["tenant_id"] = self.tenant_id

        if mapping:
            self._redis.hset(self._meta_key, mapping=mapping)
            self._redis.expire(self._meta_key, self._ttl_seconds)

    @property
    def messages(self) -> list[dict[str, Any]]:
        self._ensure_meta()

        raw_messages = self._redis.lrange(self._messages_key, 0, -1)
        parsed: list[dict[str, Any]] = []

        for item in raw_messages:
            try:
                parsed.append(json.loads(item))
            except json.JSONDecodeError:
                continue

        return parsed

    def add(self, message: dict[str, Any]) -> None:
        self._ensure_meta()
        self._redis.rpush(self._messages_key, json.dumps(message))
        self._redis.expire(self._messages_key, self._ttl_seconds)

    def reset(self) -> None:
        self._redis.delete(self._messages_key)

    def close(self) -> None:
        if self._owns_redis_client:
            self._redis.close()

    def for_conversation(self, conversation_id: str) -> MemoryBackend:
        return RedisMemory(
            redis_url=self._redis_url,
            conversation_id=conversation_id,
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            ttl_seconds=self._ttl_seconds,
            _redis_client=self._redis,
        )
