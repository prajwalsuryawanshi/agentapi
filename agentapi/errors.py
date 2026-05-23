"""Custom exceptions used by AgentAPI."""

from __future__ import annotations


class AgentAPIError(Exception):
    """Base exception for all AgentAPI errors."""


class AgentConfigurationError(AgentAPIError, ValueError):
    """Raised when AgentAPI runtime configuration is incomplete or invalid."""


class AgentProviderError(AgentAPIError, RuntimeError):
    """Raised when upstream provider API requests fail."""

    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code

    def __str__(self) -> str:
        return f"{self.status_code}: {self.message}"
