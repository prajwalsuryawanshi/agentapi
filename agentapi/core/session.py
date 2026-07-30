"""Per-request memory factory and session-scoped memory helpers.

This module addresses the shared-memory concurrency safety problem
described in issue #30.

When a single ``Agent`` instance is mounted as a FastAPI singleton
(the most common pattern), its ``memory`` attribute is shared across
all concurrent requests.  This means:\n
* User A's messages are visible in User B's conversation history.
* Concurrent ``run()`` / ``stream()`` calls mutate the same list,
  causing race conditions and interleaved history.

The solution is **request-scoped memory**: each HTTP request receives
its own ``MemoryBackend`` instance, keyed on a session / conversation
ID that the caller controls.  The ``MemoryFactory`` abstraction makes
this easy to plug in without changing the ``Agent`` API.

Usage (FastAPI dependency injection)::

    from fastapi import FastAPI, Depends, Header
    from agentapi import Agent
    from agentapi.agent.memory import InMemoryMemory
    from agentapi.core.session import MemoryFactory, per_request_memory

    agent = Agent(system_prompt="You are helpful.", provider="openai")
    app = FastAPI()

    memory_store = per_request_memory()  # one factory, shared across routes

    @app.post("/chat")
    async def chat(
        message: str,
        session_id: str = Header(default=None),
        memory = Depends(memory_store.as_dependency),
    ):
        # Agent is given a request-scoped memory; no state leak possible.
        agent.memory = memory
        return await agent.run(message)
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from typing import Any

from agentapi.agent.memory import InMemoryMemory, MemoryBackend

logger = logging.getLogger(__name__)


class MemoryFactory:
    """A thread-safe store of per-session ``MemoryBackend`` instances.

    ``MemoryFactory`` creates and caches one ``MemoryBackend`` per unique
    *session_id*.  Multiple concurrent requests that share the same
    *session_id* receive the **same** backend (enabling persistent
    multi-turn conversations), while different session IDs remain
    completely isolated from each other.

    Args:
        backend_factory: A zero-argument callable that returns a fresh
            ``MemoryBackend``.  Defaults to ``InMemoryMemory``.

    Example::

        factory = MemoryFactory()                        # InMemory default
        mem_a = factory.get("session-alice")             # Alice's memory
        mem_b = factory.get("session-bob")               # Bob's — isolated
        assert factory.get("session-alice") is mem_a     # Same object reused
    """

    def __init__(
        self,
        backend_factory: Callable[[], MemoryBackend] = InMemoryMemory,
    ) -> None:
        self._backend_factory = backend_factory
        self._sessions: dict[str, MemoryBackend] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> MemoryBackend:
        """Return the ``MemoryBackend`` for *session_id*, creating it if needed.

        This method is thread-safe: concurrent calls with the same
        *session_id* will never create duplicate backends.
        """
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = self._backend_factory()
                logger.debug("[AgentAPI] MemoryFactory: created memory for session %s", session_id)
            return self._sessions[session_id]

    def drop(self, session_id: str) -> None:
        """Evict the memory for *session_id* (e.g. on logout or session expiry)."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.debug("[AgentAPI] MemoryFactory: dropped memory for session %s", session_id)

    def active_sessions(self) -> list[str]:
        """Return a snapshot of currently active session IDs."""
        with self._lock:
            return list(self._sessions.keys())

    def as_dependency(self, session_id: str | None = None) -> MemoryBackend:
        """FastAPI-compatible dependency that resolves a per-session memory.

        Inject this as a ``Depends`` in route handlers::

            from fastapi import Depends, Header

            @app.post("/chat")
            async def chat(
                message: str,
                session_id: str = Header(default="default"),
                memory: MemoryBackend = Depends(memory_store.as_dependency),
            ):
                agent.memory = memory
                return await agent.run(message)

        When no ``session_id`` header is provided the backend falls back to
        the literal key ``"default"``, so single-user deployments work
        without any extra headers.
        """
        return self.get(session_id or "default")


def per_request_memory(
    backend_factory: Callable[[], MemoryBackend] = InMemoryMemory,
) -> MemoryFactory:
    """Create a :class:`MemoryFactory` for use as a FastAPI dependency.

    This is the recommended entry point.  Call it once at module level to
    create the factory, then inject ``factory.as_dependency`` into routes.

    Args:
        backend_factory: Optional custom ``MemoryBackend`` constructor.
            Defaults to :class:`~agentapi.agent.memory.InMemoryMemory`.

    Returns:
        A :class:`MemoryFactory` instance ready for FastAPI dependency
        injection.

    Example::

        memory_store = per_request_memory()     # call once, at module level

        @app.post("/chat")
        async def chat(
            message: str,
            session_id: str = Header(default="anonymous"),
            memory: MemoryBackend = Depends(memory_store.as_dependency),
        ):
            agent.memory = memory
            return await agent.run(message)
    """
    return MemoryFactory(backend_factory)
