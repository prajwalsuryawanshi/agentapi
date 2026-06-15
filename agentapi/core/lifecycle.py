"""Lifecycle helpers for AgentAPI provider cleanup.

This module provides :func:`create_lifespan` — an async context manager
factory that closes registered :class:`~agentapi.providers.base.BaseProvider`
instances cleanly during FastAPI app shutdown.

Usage::

    from agentapi import AgentAPI, Agent
    from agentapi.providers.openai import OpenAIProvider
    from agentapi.core.lifecycle import create_lifespan

    provider = OpenAIProvider(api_key="...", model="gpt-4o-mini")
    agent = Agent(system_prompt="...", provider=provider)

    app = AgentAPI(lifespan=create_lifespan([provider]))

    @app.chat("/chat")
    async def chat(message: str):
        return await agent.run(message)
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Any

from agentapi.providers.base import BaseProvider

logger = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def _provider_lifespan(
    app: Any,
    providers: list[BaseProvider],
) -> AsyncIterator[None]:
    """Internal lifespan that closes *providers* on shutdown."""
    try:
        yield
    finally:
        for provider in providers:
            try:
                await provider.close()
                logger.debug(
                    "[AgentAPI] Provider closed",
                    extra={"provider": type(provider).__name__},
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "[AgentAPI] Error closing provider",
                    extra={"provider": type(provider).__name__},
                )


def create_lifespan(*providers: BaseProvider):
    """Return a FastAPI-compatible lifespan context manager that closes *providers*.

    Pass the returned value as the ``lifespan`` argument to
    :class:`~agentapi.core.app.AgentAPI` (or ``fastapi.FastAPI``).

    Args:
        *providers: One or more :class:`~agentapi.providers.base.BaseProvider`
            instances to close when the application shuts down.

    Returns:
        An async context manager suitable for ``lifespan=`` in FastAPI.

    Example::

        provider = OpenAIProvider(api_key="sk-...", model="gpt-4o-mini")
        app = AgentAPI(lifespan=create_lifespan(provider))
    """
    provider_list = list(providers)

    @contextlib.asynccontextmanager
    async def lifespan(app: Any) -> AsyncIterator[None]:
        async with _provider_lifespan(app, provider_list):
            yield

    return lifespan
