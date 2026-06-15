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


class InMemoryMemory(MemoryBackend):
    """Stores chat messages in process memory with per-conversation isolation.

    Supports multiple conversations keyed by UUID. Ideal for development and
    testing multi-user scenarios without external dependencies.
    """

    def __init__(
        self,
        conversation_id: str | None = None,
    ) -> None:
        # Validate and normalize to canonical UUID string if provided; auto-generate otherwise.
        if conversation_id is not None:
            self.conversation_id = str(UUID(conversation_id))
        else:
            self.conversation_id = create_conversation_id()

        # Per-conversation message storage and system prompts.
        self._conversations: dict[str, list[dict[str, Any]]] = {}

        # Initialize this conversation.
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
    ) -> None:
        try:
            redis_module = import_module("redis")
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise ImportError("redis package is required for RedisMemory. Install with: pip install redis") from exc

        Redis = getattr(redis_module, "Redis")

        # Validate and normalize to canonical UUID string.
        self.conversation_id = str(UUID(conversation_id))
        self.user_id = user_id
        self.tenant_id = tenant_id
        self._ttl_seconds = ttl_seconds
        self._redis = Redis.from_url(redis_url, decode_responses=True)

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
        self._redis.close()


class PostgresMemory(MemoryBackend):
    """PostgreSQL-backed memory for multi-user and multi-worker deployments.

    Stores conversation messages in a ``agentapi_conversations`` table as
    JSONB rows.  The table is created automatically on first use if it does
    not already exist.

    Requires: ``pip install asyncpg``

    Example usage::

        import asyncio
        from agentapi import Agent, PostgresMemory

        async def main():
            memory = await PostgresMemory.create(
                dsn="postgresql://user:password@localhost/mydb",
                conversation_id="some-uuid",
            )
            agent = Agent(
                system_prompt="You are a helpful assistant",
                provider="openai",
                memory=memory,
            )
            reply = await agent.run("Hello!")
            print(reply)
            await memory.close()

        asyncio.run(main())
    """

    _CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS agentapi_conversations (
            id            BIGSERIAL PRIMARY KEY,
            conversation_id TEXT      NOT NULL,
            user_id         TEXT,
            tenant_id       TEXT,
            message_data    JSONB     NOT NULL,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_agentapi_conv_id
            ON agentapi_conversations (conversation_id);
    """

    def __init__(
        self,
        *,
        pool: Any,  # asyncpg.Pool
        conversation_id: str,
        user_id: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        """Low-level constructor.  Prefer :meth:`create` for normal usage."""
        # Validate and normalize to canonical UUID string.
        self.conversation_id = str(UUID(conversation_id))
        self.user_id = user_id
        self.tenant_id = tenant_id
        self._pool = pool

    # ------------------------------------------------------------------
    # Async factory
    # ------------------------------------------------------------------

    @classmethod
    async def create(
        cls,
        *,
        dsn: str,
        conversation_id: str,
        user_id: str | None = None,
        tenant_id: str | None = None,
        min_size: int = 1,
        max_size: int = 10,
    ) -> "PostgresMemory":
        """Create a connection pool and return a ready-to-use ``PostgresMemory``.

        Args:
            dsn: PostgreSQL connection string, e.g.
                ``"postgresql://user:password@host/dbname"``.
            conversation_id: UUID string that identifies this conversation.
            user_id: Optional owner identifier for multi-tenant isolation.
            tenant_id: Optional tenant identifier for multi-tenant isolation.
            min_size: Minimum connections kept in the pool (default 1).
            max_size: Maximum connections in the pool (default 10).

        Returns:
            A fully initialised :class:`PostgresMemory` instance.

        Raises:
            ImportError: If ``asyncpg`` is not installed.
        """
        try:
            asyncpg = import_module("asyncpg")
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "asyncpg package is required for PostgresMemory. "
                "Install with: pip install asyncpg"
            ) from exc

        pool = await asyncpg.create_pool(dsn, min_size=min_size, max_size=max_size)
        instance = cls(
            pool=pool,
            conversation_id=conversation_id,
            user_id=user_id,
            tenant_id=tenant_id,
        )
        await instance._ensure_table()
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_table(self) -> None:
        """Create the conversation table if it does not exist."""
        async with self._pool.acquire() as conn:
            await conn.execute(self._CREATE_TABLE_SQL)

    # ------------------------------------------------------------------
    # MemoryBackend interface
    # ------------------------------------------------------------------

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Synchronous property required by MemoryBackend.

        .. note::
            This performs a blocking database fetch via
            :meth:`asyncio.get_event_loop().run_until_complete`.  In
            async contexts, prefer awaiting :meth:`async_messages` directly.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.async_messages())

    async def async_messages(self) -> list[dict[str, Any]]:
        """Async variant of :attr:`messages`; use this inside async code."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT message_data
                  FROM agentapi_conversations
                 WHERE conversation_id = $1
                 ORDER BY id ASC
                """,
                self.conversation_id,
            )
        return [json.loads(row["message_data"]) for row in rows]

    def add(self, message: dict[str, Any]) -> None:
        """Synchronous add required by MemoryBackend.

        Delegates to :meth:`async_add` via the event loop.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_add(message))

    async def async_add(self, message: dict[str, Any]) -> None:
        """Async variant of :meth:`add`; use this inside async code."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agentapi_conversations
                    (conversation_id, user_id, tenant_id, message_data)
                VALUES ($1, $2, $3, $4::jsonb)
                """,
                self.conversation_id,
                self.user_id,
                self.tenant_id,
                json.dumps(message),
            )

    def reset(self) -> None:
        """Synchronous reset required by MemoryBackend.

        Delegates to :meth:`async_reset` via the event loop.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_reset())

    async def async_reset(self) -> None:
        """Async variant of :meth:`reset`; use this inside async code."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM agentapi_conversations WHERE conversation_id = $1",
                self.conversation_id,
            )

    async def close(self) -> None:
        """Close the underlying connection pool.

        Call this when the application shuts down to release database
        connections cleanly.
        """
        await self._pool.close()
