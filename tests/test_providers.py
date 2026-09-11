"""Unit tests for ProviderResponse token usage tracking."""

import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from agentapi.providers.base import ProviderResponse
from agentapi.observability import TokenUsage, safe_int_usage
from agentapi.providers.openai_compatible import OpenAICompatibleProvider
from agentapi.providers.gemini import GeminiProvider
from agentapi.providers.anthropic import AnthropicProvider


def test_safe_int_usage():
    assert safe_int_usage(10) == 10
    assert safe_int_usage("20") == 20
    assert safe_int_usage(15.0) == 15
    assert safe_int_usage(None, default=5) == 5
    assert safe_int_usage("invalid", default=0) == 0
    assert safe_int_usage(float("inf"), default=0) == 0
    assert safe_int_usage(float("nan"), default=0) == 0
    assert safe_int_usage(-5, default=0) == 0
    assert safe_int_usage(True, default=0) == 0
    assert safe_int_usage(False, default=0) == 0


def test_token_usage_model():
    # Default instantiation
    usage_default = TokenUsage()
    assert usage_default.prompt_tokens == 0
    assert usage_default.completion_tokens == 0
    assert usage_default.total_tokens == 0

    # Automatic total calculation
    usage1 = TokenUsage(prompt_tokens=10, completion_tokens=20)
    assert usage1.prompt_tokens == 10
    assert usage1.completion_tokens == 20
    assert usage1.total_tokens == 30

    # Explicit total override
    usage2 = TokenUsage(prompt_tokens=15, completion_tokens=25, total_tokens=40)
    assert usage2.prompt_tokens == 15
    assert usage2.completion_tokens == 25
    assert usage2.total_tokens == 40
    assert repr(usage2) == "TokenUsage(prompt_tokens=15, completion_tokens=25, total_tokens=40)"

    # Equality and inequality
    assert usage2 == TokenUsage(15, 25, 40)
    assert usage2 != usage1
    assert usage2 != "not a TokenUsage object"


def test_openai_compatible_usage_extraction():
    async def _test():
        provider = OpenAICompatibleProvider(
            api_key="test-key",
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
        )

        mock_json = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello world",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 25,
                "total_tokens": 40,
            },
        }

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=mock_json)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            res = await provider.chat([{"role": "user", "content": "Hi"}])

            assert isinstance(res, ProviderResponse)
            assert res.content == "Hello world"
            assert res.usage == TokenUsage(
                prompt_tokens=15,
                completion_tokens=25,
                total_tokens=40,
            )

    asyncio.run(_test())


def test_openai_compatible_null_usage_keys():
    async def _test():
        provider = OpenAICompatibleProvider(
            api_key="test-key",
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
        )

        mock_json = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            },
        }

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=mock_json)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            res = await provider.chat([{"role": "user", "content": "Hi"}])

            assert isinstance(res, ProviderResponse)
            assert res.usage == TokenUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )

    asyncio.run(_test())


def test_openai_compatible_usage_missing():
    async def _test():
        provider = OpenAICompatibleProvider(
            api_key="test-key",
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
        )

        mock_json = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello",
                    }
                }
            ]
        }

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=mock_json)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            res = await provider.chat([{"role": "user", "content": "Hi"}])

            assert isinstance(res, ProviderResponse)
            assert res.usage is None

    asyncio.run(_test())


def test_gemini_usage_extraction():
    async def _test():
        provider = GeminiProvider(
            api_key="test-key",
            model="gemini-2.5-flash",
        )

        mock_json = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Gemini answer"}]
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 50,
                "totalTokenCount": 150,
            },
        }

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=mock_json)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            res = await provider.chat([{"role": "user", "content": "Hi"}])

            assert isinstance(res, ProviderResponse)
            assert res.content == "Gemini answer"
            assert res.usage == TokenUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            )

    asyncio.run(_test())


def test_gemini_usage_missing():
    async def _test():
        provider = GeminiProvider(
            api_key="test-key",
            model="gemini-2.5-flash",
        )

        mock_json = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Gemini answer"}]
                    }
                }
            ]
        }

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=mock_json)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            res = await provider.chat([{"role": "user", "content": "Hi"}])

            assert isinstance(res, ProviderResponse)
            assert res.usage is None

    asyncio.run(_test())


def test_anthropic_usage_extraction_with_cache():
    async def _test():
        with patch("agentapi.providers.anthropic.AsyncAnthropic"):
            provider = AnthropicProvider(api_key="test-key", model="claude-3-5-sonnet-20241022")

            mock_block = MagicMock()
            mock_block.type = "text"
            mock_block.text = "Claude response"

            mock_usage = MagicMock(
                spec=[
                    "input_tokens",
                    "output_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                ]
            )
            mock_usage.input_tokens = 40
            mock_usage.output_tokens = 20
            mock_usage.cache_creation_input_tokens = 10
            mock_usage.cache_read_input_tokens = 5

            mock_response = MagicMock()
            mock_response.content = [mock_block]
            mock_response.usage = mock_usage
            mock_response.model_dump = MagicMock(return_value={
                "content": [{"type": "text", "text": "Claude response"}],
                "usage": {
                    "input_tokens": 40,
                    "output_tokens": 20,
                    "cache_creation_input_tokens": 10,
                    "cache_read_input_tokens": 5,
                }
            })

            provider.client.messages.create = AsyncMock(return_value=mock_response)

            res = await provider.chat([{"role": "user", "content": "Hi"}])

            assert isinstance(res, ProviderResponse)
            assert res.content == "Claude response"
            assert res.usage == TokenUsage(
                prompt_tokens=55,  # 40 + 10 + 5
                completion_tokens=20,
                total_tokens=75,
            )

    asyncio.run(_test())


def test_anthropic_usage_missing():
    async def _test():
        with patch("agentapi.providers.anthropic.AsyncAnthropic"):
            provider = AnthropicProvider(api_key="test-key", model="claude-3-5-sonnet-20241022")

            mock_block = MagicMock()
            mock_block.type = "text"
            mock_block.text = "Claude response"

            mock_response = MagicMock()
            mock_response.content = [mock_block]
            mock_response.usage = None

            provider.client.messages.create = AsyncMock(return_value=mock_response)

            res = await provider.chat([{"role": "user", "content": "Hi"}])

            assert isinstance(res, ProviderResponse)
            assert res.usage is None

    asyncio.run(_test())
