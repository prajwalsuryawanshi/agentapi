"""Observability module for tracking model usage and telemetry."""

from __future__ import annotations

__all__ = ["TokenUsage", "safe_int_usage", "RawUsageValue"]

RawUsageValue = int | float | str | None


class TokenUsage:
    """Normalized token usage tracking for model responses."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __init__(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int | None = None,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = (
            total_tokens
            if total_tokens is not None
            else (prompt_tokens + completion_tokens)
        )

    def __repr__(self) -> str:
        return (
            f"TokenUsage(prompt_tokens={self.prompt_tokens}, "
            f"completion_tokens={self.completion_tokens}, "
            f"total_tokens={self.total_tokens})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TokenUsage):
            return False
        return (
            self.prompt_tokens == other.prompt_tokens
            and self.completion_tokens == other.completion_tokens
            and self.total_tokens == other.total_tokens
        )


def safe_int_usage(val: RawUsageValue, default: int = 0) -> int:
    """Safely coerce raw provider usage metadata values to a non-negative integer.

    Accepts numeric values and numeric strings. Rejects None, booleans,
    negative counts, and non-numeric values, returning default on error or overflow.
    """
    if val is None or isinstance(val, bool):
        return default
    try:
        num = int(val)
        return num if num >= 0 else default
    except (ValueError, TypeError, OverflowError):
        return default
