"""Custom exceptions used by AgentAPI."""

from __future__ import annotations


class AgentConfigurationError(ValueError):
    """Raised when AgentAPI runtime configuration is incomplete or invalid."""


class AgentProviderError(RuntimeError):
    """Raised when upstream provider API requests fail."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class AgentSchemaValidationError(RuntimeError):
    """Raised when an LLM fails to output valid JSON matching a Pydantic schema."""
    pass
