"""Provider interface and shared response models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator


@dataclass
class ToolCall:
    """A tool call emitted by the model."""

    id: str
    name: str
    arguments: str


@dataclass
class ProviderResponse:
    """Normalized non-stream model response."""

    content: str
    tool_calls: list[ToolCall]
    raw_message: dict[str, Any]


class BaseProvider(ABC):
    """Abstract provider contract used by Agent.

    Concrete providers should accept an optional *client* keyword argument
    (``httpx.AsyncClient``) in their ``__init__``.  When a client is supplied,
    the provider reuses it for every request, which enables connection pooling
    and avoids socket-churn across calls.  When no client is supplied, the
    provider creates and **owns** its own client and is responsible for closing
    it in :meth:`close`.

    The recommended pattern is to let :class:`~agentapi.core.app.AgentAPI`
    manage provider lifecycle automatically via its ``lifespan`` context
    manager.  See :meth:`agentapi.core.app.AgentAPI.with_lifespan` for details.
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_calling: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        pass

    @abstractmethod
    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_calling: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        pass

    async def close(self) -> None:
        """Release resources held by this provider (e.g. the HTTP client).

        The default implementation is a no-op.  Concrete providers that own an
        ``httpx.AsyncClient`` **must** override this and call
        ``await self._client.aclose()``.

        This method is called automatically by
        :class:`~agentapi.core.app.AgentAPI` during app shutdown when the
        provider was registered via
        :meth:`~agentapi.core.app.AgentAPI.register_provider_for_cleanup`.
        """
