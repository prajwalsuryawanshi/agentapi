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
    """Abstract provider contract used by Agent."""

    def default_tool_calling(self) -> dict[str, Any]:
        """Return provider-native defaults for tool/function calling."""

        return {}

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
