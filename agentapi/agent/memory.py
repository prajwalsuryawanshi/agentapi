"""Conversation memory backends for agents."""

from __future__ import annotations

import os
import json
import sqlite3
from importlib import import_module
from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID, uuid4
from pathlib import Path


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


class FileMemory(MemoryBackend):
    """Stores chat messages in local JSON files.

    Provides lightweight persistence across application restarts without 
    requiring external databases like Redis.
    """

    def __init__(
        self,
        conversation_id: str | None = None,
        *,
        storage_dir: str | Path = ".agent_memory",
    ) -> None:
        if conversation_id is not None:
            self.conversation_id = str(UUID(conversation_id))
        else:
            self.conversation_id = create_conversation_id()
            
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.storage_dir / f"{self.conversation_id}.json"

    @property
    def messages(self) -> list[dict[str, Any]]:
        if not self.file_path.exists():
            return []
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

    def add(self, message: dict[str, Any]) -> None:
        messages = self.messages
        messages.append(message)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2)

    def reset(self) -> None:
        if self.file_path.exists():
            self.file_path.unlink()


class SqliteMemory(MemoryBackend):
    """SQLite-backed memory for robust local persistence.

    Uses Python's built-in sqlite3 module to store conversations in a single
    database file, allowing for easy querying and backup.
    """

    def __init__(
        self,
        conversation_id: str | None = None,
        *,
        db_path: str | Path = ".agent_memory/memory.db",
    ) -> None:
        if conversation_id is not None:
            self.conversation_id = str(UUID(conversation_id))
        else:
            self.conversation_id = create_conversation_id()
            
        # Ensure directory exists
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                message_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_conv_id 
            ON messages (conversation_id)
        ''')
        self.conn.commit()

    @property
    def messages(self) -> list[dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT message_json FROM messages WHERE conversation_id = ? ORDER BY id ASC", 
            (self.conversation_id,)
        )
        rows = cursor.fetchall()
        
        parsed = []
        for row in rows:
            try:
                parsed.append(json.loads(row[0]))
            except json.JSONDecodeError:
                continue
        return parsed

    def add(self, message: dict[str, Any]) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO messages (conversation_id, message_json) VALUES (?, ?)",
            (self.conversation_id, json.dumps(message))
        )
        self.conn.commit()

    def reset(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM messages WHERE conversation_id = ?",
            (self.conversation_id,)
        )
        self.conn.commit()
        
    def close(self) -> None:
        self.conn.close()


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
