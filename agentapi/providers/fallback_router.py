
from __future__ import annotations
import logging
from typing import Any, AsyncIterator
from agentapi.providers.base import BaseProvider, ProviderResponse

logger = logging.getLogger(__name__)


async def chat_with_fallback(
    providers: list[BaseProvider],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_calling: dict[str, Any] | None,
    max_retries: int = 1,
) -> ProviderResponse:
    """
    Primary provider se chat karo.
    Agar HTTP 429 (rate limit) ya 500+ (server error) aaye,
    to next provider try karo — max_retries baar tak.
    """
    last_exc = None

    for attempt, provider in enumerate(providers):
        retries = 0
        while retries <= max_retries:
            try:
                logger.info(
                    f"[Fallback] Trying provider {provider.__class__.__name__} "
                    f"(attempt {attempt + 1}, retry {retries})"
                )
                return await provider.chat(
                    messages,
                    tools=tools,
                    tool_calling=tool_calling,
                )
            except Exception as exc:
                last_exc = exc
                
                if _is_retriable(exc) and retries < max_retries:
                    retries += 1
                    logger.warning(
                        f"[Fallback] Provider {provider.__class__.__name__} "
                        f"failed ({exc}). Retrying {retries}/{max_retries}..."
                    )
                else:
                    
                    logger.warning(
                        f"[Fallback] Provider {provider.__class__.__name__} "
                        f"exhausted. Moving to next provider. Error: {exc}"
                    )
                    break

    raise Exception(
        f"All providers failed. Last error: {last_exc}"
    )


async def stream_with_fallback(
    providers: list[BaseProvider],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_calling: dict[str, Any] | None,
) -> AsyncIterator[str]:
    """
    Stream tokens — agar primary fail ho to next provider se stream karo.
    """
    last_exc = None

    for provider in providers:
        try:
            logger.info(
                f"[Fallback] Streaming with {provider.__class__.__name__}"
            )
            async for token in provider.stream(
                messages,
                tools=tools,
                tool_calling=tool_calling,
            ):
                yield token
            return  # bahar nikal jao

        except Exception as exc:
            last_exc = exc
            logger.warning(
                f"[Fallback] Stream failed for {provider.__class__.__name__}: {exc}. "
                f"Trying next provider..."
            )

    raise Exception(f"All providers failed during streaming. Last error: {last_exc}")


def _is_retriable(exc: Exception) -> bool:
    """
    Check karo ki yeh error retry karne layak hai ya nahi.
    HTTP 429 = rate limit → retry
    HTTP 500+ = server error → retry
    """
    msg = str(exc).lower()
    # Common rate limit or server error patterns
    return any(keyword in msg for keyword in [
        "429", "rate limit", "too many requests",
        "500", "502", "503", "server error", "overloaded",
        "timeout", "connection error"
    ])