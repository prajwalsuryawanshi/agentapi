"""Resilient streaming utilities for AgentAPI.

This module provides :class:`ResilientStream` — a wrapper around
:meth:`~agentapi.agent.agent.Agent.astream` that adds configurable retry
logic and a clean fallback signal when all retries are exhausted.

Usage::

    from agentapi.core.stream import ResilientStream

    # Using as an async context manager
    async with ResilientStream(agent, "What is the capital of France?") as stream:
        async for chunk in stream:
            print(chunk, end="", flush=True)

    # Direct iteration (also works)
    async for chunk in ResilientStream(agent, "hello"):
        print(chunk)
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, AsyncIterator

if TYPE_CHECKING:
    from agentapi.agent.agent import Agent

logger = logging.getLogger(__name__)


class StreamError(Exception):
    """Raised when all retry attempts for a stream are exhausted."""

    def __init__(self, message: str, *, attempt_count: int, last_error: Exception) -> None:
        super().__init__(message)
        self.attempt_count = attempt_count
        self.last_error = last_error


class ResilientStream:
    """A resilient wrapper around :meth:`~agentapi.agent.agent.Agent.astream`.

    When the underlying provider stream fails mid-flight (due to network
    drops, rate limits, or transient server errors), :class:`ResilientStream`
    automatically retries up to *max_retries* times with an exponential
    back-off delay between attempts.

    If all retries fail, a structured ``[STREAM_ERROR]`` sentinel token is
    yielded so that downstream consumers (e.g. SSE clients) receive a
    machine-readable signal instead of a silent hang or crash.  After the
    sentinel, a :class:`StreamError` is raised.

    The class implements both the async iterator protocol and the async
    context manager protocol, so it can be used with or without ``async with``.

    Args:
        agent: The :class:`~agentapi.agent.agent.Agent` instance to stream from.
        message: The user message to send.
        max_retries: Maximum number of additional attempts after the first
            failure (default ``2``, so up to 3 total attempts).
        backoff_seconds: Base delay in seconds between retries (default
            ``1.0``).  Each subsequent retry doubles the delay (exponential
            back-off) up to a cap of 30 seconds.
        error_sentinel: The string token emitted before raising
            :class:`StreamError` when all retries are exhausted (default
            ``"[STREAM_ERROR] Stream failed after retries. Check server logs."``).  Set to
            ``None`` to suppress the sentinel.

    Example::

        async for chunk in ResilientStream(agent, "hello", max_retries=3):
            print(chunk, end="")

        # Or, if you want to suppress the error sentinel and handle the
        # StreamError yourself:
        try:
            async for chunk in ResilientStream(
                agent, "hello",
                error_sentinel=None,
            ):
                print(chunk, end="")
        except StreamError as exc:
            print(f"Failed after {exc.attempt_count} attempts: {exc.last_error}")
    """

    ERROR_SENTINEL = "[STREAM_ERROR] Stream failed after retries. Check server logs."

    def __init__(
        self,
        agent: "Agent",
        message: str,
        *,
        max_retries: int = 2,
        backoff_seconds: float = 1.0,
        error_sentinel: str | None = None,
    ) -> None:
        self._agent = agent
        self._message = message
        self._max_retries = max(0, max_retries)
        self._backoff_seconds = max(0.0, backoff_seconds)
        self._error_sentinel = error_sentinel if error_sentinel is not None else self.ERROR_SENTINEL
        self._aiter: AsyncIterator[str] | None = None

    def __aiter__(self) -> "ResilientStream":
        return self

    async def __anext__(self) -> str:
        # Lazily initialise the underlying generator on first __anext__ call.
        if self._aiter is None:
            self._aiter = self._retry_generator()
        try:
            return await self._aiter.__anext__()
        except StopAsyncIteration:
            raise

    async def __aenter__(self) -> "ResilientStream":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """No-op exit; resources are managed by the underlying generator."""

    async def _retry_generator(self) -> AsyncIterator[str]:
        """Internal generator with retry logic."""
        last_error: Exception | None = None
        total_attempts = self._max_retries + 1

        for attempt in range(total_attempts):
            if attempt > 0:
                delay = min(self._backoff_seconds * (2 ** (attempt - 1)), 30.0)
                logger.warning(
                    "[AgentAPI] ResilientStream: retrying stream attempt %d/%d after %.1fs. Error: %s",
                    attempt + 1,
                    total_attempts,
                    delay,
                    last_error,
                )
                await asyncio.sleep(delay)

            try:
                async for token in self._agent.astream(self._message):
                    yield token
                # Successful completion: stop retrying.
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "[AgentAPI] ResilientStream: stream attempt %d/%d failed: %s",
                    attempt + 1,
                    total_attempts,
                    exc,
                )

        # All attempts exhausted.
        logger.error(
            "[AgentAPI] ResilientStream: all %d attempt(s) failed. Last error: %s",
            total_attempts,
            last_error,
        )
        if self._error_sentinel:
            yield self._error_sentinel
        raise StreamError(
            f"Stream failed after {total_attempts} attempt(s)",
            attempt_count=total_attempts,
            last_error=last_error,  # type: ignore[arg-type]
        )
