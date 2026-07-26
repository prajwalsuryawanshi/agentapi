"""Conversation memory backends for agents."""

from __future__ import annotations

import json
from importlib import import_module
from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID, uuid4
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

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


# Motor and Redis are optional dependencies — imported lazily so users who
# don't need Mongo/Redis don't have to install them.
# Install:  pip install "agentapi-core[mongo]"
#           pip install "agentapi-core[mongo,redis]"  ← for Redis cache flag
try:
    from motor.motor_asyncio import AsyncIOMotorClient
    _MOTOR_AVAILABLE = True
except ImportError:
    _MOTOR_AVAILABLE = False

try:
    import redis.asyncio as aioredis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


class MongoMemory:
    """
    MongoDB-backed conversation memory for AgentAPI.

    Stores per-session conversation history as a document in a MongoDB
    collection. Each document is keyed by ``session_id`` and holds the
    full ordered list of messages.

    Production mode (``use_redis_cache=True``):
        Adds a Redis write-through cache layer in front of MongoDB.
        - Reads hit Redis first (microsecond latency).
        - Writes go to both Redis and MongoDB atomically.
        - Redis entries expire automatically via ``redis_ttl`` (default 1 hour).
        - On cache miss, falls back to MongoDB and repopulates Redis.
        Requires: ``pip install "agentapi-core[mongo,redis]"``

    Development mode (``use_redis_cache=False``, default):
        MongoDB only. No Redis dependency required.
        Requires: ``pip install "agentapi-core[mongo]"``

    Args:
        mongo_uri:         MongoDB connection string.
                           e.g. ``"mongodb://localhost:27017"``
        db_name:           MongoDB database name. Default: ``"agentapi"``.
        collection_name:   MongoDB collection name. Default: ``"conversations"``.
        session_id:        Unique identifier for this conversation session.
                           Use one per user/chat thread. Default: ``"default"``.
        max_history:       If set, only the most recent N messages are kept.
                           Older messages are pruned on every write.
                           Default: ``None`` (unlimited).
        use_redis_cache:   Enable Redis as a write-through cache layer.
                           Set ``True`` for production deployments.
                           Default: ``False``.
        redis_url:         Redis connection URL. Required when
                           ``use_redis_cache=True``.
                           e.g. ``"redis://localhost:6379"``
        redis_ttl:         Redis key TTL in seconds. Default: ``3600`` (1 hour).
                           After this window with no activity, the Redis cache
                           entry expires; the next read falls back to MongoDB.

    Example — development (MongoDB only)::

        from agentapi.agent.memory import MongoMemory

        memory = MongoMemory(
            mongo_uri="mongodb://localhost:27017",
            session_id="user_42",
        )
        agent = Agent(provider="openai", memory=memory)

    Example — production (MongoDB + Redis cache)::

        memory = MongoMemory(
            mongo_uri="mongodb://prod-mongo:27017",
            session_id="user_42",
            use_redis_cache=True,
            redis_url="redis://prod-redis:6379",
            redis_ttl=3600,
        )
        agent = Agent(provider="openai", memory=memory)
    """

    # Redis key prefix — avoids collisions with other Redis users
    _REDIS_KEY_PREFIX = "agentapi:memory:"

    def __init__(
        self,
        mongo_uri: str,
        db_name: str = "agentapi",
        collection_name: str = "conversations",
        session_id: str = "default",
        max_history: Optional[int] = None,
        *,
        use_redis_cache: bool = False,
        redis_url: Optional[str] = None,
        redis_ttl: int = 3600,
    ) -> None:
        # ── Validate dependencies at construction time ──────────────────────
        if not _MOTOR_AVAILABLE:
            raise ImportError(
                "MongoMemory requires 'motor'. "
                "Install it with: pip install \"agentapi-core[mongo]\""
            )

        if use_redis_cache and not _REDIS_AVAILABLE:
            raise ImportError(
                "use_redis_cache=True requires 'redis[asyncio]'. "
                "Install it with: pip install \"agentapi-core[mongo,redis]\""
            )

        if use_redis_cache and not redis_url:
            raise ValueError(
                "use_redis_cache=True requires redis_url to be provided. "
                "Example: redis_url=\"redis://localhost:6379\""
            )

        # ── MongoDB setup ────────────────────────────────────────────────────
        self._mongo_client = AsyncIOMotorClient(mongo_uri)
        self._collection = self._mongo_client[db_name][collection_name]
        self._session_id = session_id
        self._max_history = max_history

        # ── Redis cache setup ────────────────────────────────────────────────
        self._use_redis_cache: bool = use_redis_cache
        self._redis_ttl: int = redis_ttl
        self._redis: Optional[aioredis.Redis] = None  # type: ignore[type-arg]

        if use_redis_cache:
            # Lazy async connection — opened on first use
            self._redis = aioredis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
            )

        logger.debug(
            "MongoMemory initialised — session=%s redis_cache=%s",
            session_id,
            use_redis_cache,
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    @property
    def _redis_key(self) -> str:
        """Namespaced Redis key for this session's message list."""
        return f"{self._REDIS_KEY_PREFIX}{self._session_id}"

    async def _redis_get(self) -> Optional[list[dict]]:
        """
        Attempt to read the message list from Redis.
        Returns None on cache miss or any Redis error (fail open).
        """
        if not self._use_redis_cache or self._redis is None:
            return None
        try:
            raw = await self._redis.get(self._redis_key)
            if raw is None:
                return None  # Cache miss
            return json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "MongoMemory: Redis read failed, falling back to MongoDB: %s", exc
            )
            return None

    async def _redis_set(self, messages: list[dict]) -> None:
        """
        Write the full message list to Redis with TTL refresh.
        Fails silently — a Redis write error must not break the agent.
        """
        if not self._use_redis_cache or self._redis is None:
            return
        try:
            await self._redis.set(
                self._redis_key,
                json.dumps(messages),
                ex=self._redis_ttl,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "MongoMemory: Redis write failed (MongoDB write succeeded): %s", exc
            )

    async def _redis_delete(self) -> None:
        """Delete this session's Redis cache entry."""
        if not self._use_redis_cache or self._redis is None:
            return
        try:
            await self._redis.delete(self._redis_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MongoMemory: Redis delete failed: %s", exc)

    # ── Public interface ──────────────────────────────────────────────────────

    async def add(self, message: dict) -> None:
        """
        Append a message to this session's conversation history.

        Writes to MongoDB first (durable), then updates Redis cache.
        If ``max_history`` is set, older messages beyond the limit are
        pruned atomically in MongoDB using ``$push`` with ``$slice``.

        Args:
            message: A message dict, e.g.
                     ``{"role": "user", "content": "Hello"}``
        """
        push_op: dict = {"$each": [message]}

        if self._max_history is not None:
            # $slice with a negative value keeps the LAST N elements.
            # This trims the array to max_history atomically on the DB side.
            push_op["$slice"] = -self._max_history

        await self._collection.update_one(
            {"session_id": self._session_id},
            {"$push": {"messages": push_op}},
            upsert=True,
        )

        # Keep Redis in sync — fetch the current state from Mongo and cache it
        if self._use_redis_cache:
            messages = await self._mongo_get()
            await self._redis_set(messages)

    async def _mongo_get(self) -> list[dict]:
        """Fetch message list directly from MongoDB (bypasses Redis)."""
        doc = await self._collection.find_one(
            {"session_id": self._session_id},
            {"_id": 0, "messages": 1},
        )
        if doc is None:
            return []
        return doc.get("messages", [])

    async def get(self) -> list[dict]:
        """
        Return the full conversation history for this session.

        Read path (when ``use_redis_cache=True``):
            1. Try Redis (fast, sub-millisecond).
            2. On cache miss or Redis error → read from MongoDB.
            3. Repopulate Redis from MongoDB result.

        Read path (when ``use_redis_cache=False``):
            Read directly from MongoDB.

        Returns:
            List of message dicts in chronological order.
        """
        # ── Try Redis cache first ────────────────────────────────────────────
        cached = await self._redis_get()
        if cached is not None:
            logger.debug(
                "MongoMemory: cache hit for session=%s (%d messages)",
                self._session_id,
                len(cached),
            )
            return cached

        # ── Cache miss or Redis disabled → read from MongoDB ─────────────────
        messages = await self._mongo_get()

        # Repopulate Redis on cache miss
        if self._use_redis_cache and messages:
            await self._redis_set(messages)

        return messages

    async def clear(self) -> None:
        """
        Delete all conversation history for this session.

        Removes the MongoDB document AND the Redis cache entry so both
        stores are consistent after a clear.
        """
        await self._collection.delete_one({"session_id": self._session_id})
        await self._redis_delete()
        logger.debug("MongoMemory: cleared session=%s", self._session_id)

    async def close(self) -> None:
        """
        Close underlying connections gracefully.

        Call this during application shutdown to avoid resource leaks.
        Not required for short-lived scripts.
        """
        self._mongo_client.close()
        if self._redis is not None:
            await self._redis.aclose()