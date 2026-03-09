"""Custom exceptions used by AgentAPI."""

from __future__ import annotations


class AgentConfigurationError(ValueError):
    """Raised when AgentAPI runtime configuration is incomplete or invalid."""


class AgentProviderError(RuntimeError):
    """Raised when upstream provider API requests fail."""

    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code
