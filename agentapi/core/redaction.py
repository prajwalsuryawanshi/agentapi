"""Redact sensitive data from exception text before it reaches clients.

Default patterns cover common provider API key formats, bearer tokens, and
URLs with embedded credentials. Users can supply additional patterns or
disable redaction entirely via AgentApp configuration.
"""
from __future__ import annotations

import re
from typing import Iterable, Pattern


REDACTED = "[REDACTED]"

DEFAULT_PATTERNS: tuple[str, ...] = (
    # OpenAI-style keys: sk- followed by 20+ alphanumeric chars
    r"sk-[A-Za-z0-9_-]{20,}",
    # Groq-style keys: gsk_ followed by 20+ chars
    r"gsk_[A-Za-z0-9_-]{20,}",
    # Anthropic-style keys: sk-ant- followed by alphanumeric
    r"sk-ant-[A-Za-z0-9_-]{20,}",
    # Google AI Studio keys: AIza followed by 35 chars
    r"AIza[A-Za-z0-9_-]{35}",
    # Authorization header values
    r"(?i)Bearer\s+[A-Za-z0-9._~+/=-]+",
    # URLs with userinfo (e.g. https://user:pass@host)
    r"https?://[^\s:/@]+:[^\s@]+@[^\s]+",
)


def _compile_patterns(patterns: Iterable[str]) -> list[Pattern[str]]:
    return [re.compile(p) for p in patterns]


def redact(
    text: str,
    *,
    patterns: Iterable[str] | None = None,
    replacement: str = REDACTED,
) -> str:
    """Apply redaction patterns to text, returning the redacted string.

    Args:
        text: Source text, typically an exception message.
        patterns: Regex patterns to redact. Defaults to DEFAULT_PATTERNS.
        replacement: String to substitute for each match.

    Returns:
        The text with all matches replaced.
    """
    compiled = _compile_patterns(patterns if patterns is not None else DEFAULT_PATTERNS)
    result = text
    for pat in compiled:
        result = pat.sub(replacement, result)
    return result


def redact_exception(
    exc: BaseException,
    *,
    patterns: Iterable[str] | None = None,
    replacement: str = REDACTED,
) -> str:
    """Convenience wrapper: redact the string form of an exception."""
    return redact(str(exc), patterns=patterns, replacement=replacement)