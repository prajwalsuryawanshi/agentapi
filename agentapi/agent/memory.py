"""In-memory conversation storage for an agent instance."""

from __future__ import annotations

from typing import Any


class ConversationMemory:
    """Stores chat messages in a simple in-memory list."""

    def __init__(self, system_prompt: str | None = None) -> None:
        self._messages: list[dict[str, Any]] = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self._messages

    def add(self, message: dict[str, Any]) -> None:
        self._messages.append(message)

    def reset(self, system_prompt: str | None = None) -> None:
        self._messages = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})
