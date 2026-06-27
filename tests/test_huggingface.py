"""Test Hugging Face provider implementation."""

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import httpx
from agentapi.providers.huggingface import HuggingFaceProvider
from agentapi.providers.base import ProviderResponse, ToolCall
from agentapi.errors import AgentProviderError, AgentConfigurationError


def test_huggingface_init_default():
    """Test HuggingFaceProvider initializes with correct defaults."""
    provider = HuggingFaceProvider(api_key="hf_test_token", model="Qwen/Qwen2.5-72B-Instruct")
    assert provider.api_key == "hf_test_token"
    assert provider.model == "Qwen/Qwen2.5-72B-Instruct"
    assert provider.base_url == "https://api-inference.huggingface.co/v1"


def test_huggingface_init_custom_url():
    """Test HuggingFaceProvider accepts a custom base URL."""
    custom_url = "https://custom-endpoint.huggingface.cloud/v1"
    provider = HuggingFaceProvider(
        api_key="hf_test_token",
        model="custom-model",
        base_url=custom_url,
    )
    assert provider.base_url == "https://custom-endpoint.huggingface.cloud/v1"


def test_huggingface_empty_key():
    """Test that missing API key raises an error."""
    with pytest.raises(AgentConfigurationError):
        HuggingFaceProvider(api_key="", model="Qwen/Qwen2.5-72B-Instruct")


@pytest.mark.anyio
async def test_huggingface_chat_success():
    """Test successful non-stream chat request."""
    provider = HuggingFaceProvider(api_key="hf_test_token", model="Qwen/Qwen2.5-72B-Instruct")

    mock_response_data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello, this is Hugging Face!",
                }
            }
        ]
    }

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_response_data

    # Mock the AsyncClient post method
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        response = await provider.chat([{"role": "user", "content": "Hi"}])

        assert isinstance(response, ProviderResponse)
        assert response.content == "Hello, this is Hugging Face!"
        assert response.tool_calls == []
        mock_post.assert_called_once()
        # Verify correct authorization header
        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer hf_test_token"


@pytest.mark.anyio
async def test_huggingface_stream_success():
    """Test successful streaming chat request."""
    provider = HuggingFaceProvider(api_key="hf_test_token", model="Qwen/Qwen2.5-72B-Instruct")

    # Mock line iterations
    mock_lines = [
        b'data: {"choices": [{"delta": {"content": "Hello"}}]}',
        b'data: {"choices": [{"delta": {"content": " world"}}]}',
        b"data: [DONE]",
    ]

    class MockAsyncIterator:
        def __init__(self, items):
            self.items = items

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.items:
                raise StopAsyncIteration
            return self.items.pop(0).decode("utf-8")

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.aiter_lines.return_value = MockAsyncIterator(mock_lines)

    # Mock the client.stream context manager
    class MockContextManager:
        async def __aenter__(self):
            return mock_response

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch("httpx.AsyncClient.stream") as mock_stream:
        mock_stream.return_value = MockContextManager()

        tokens = []
        async for token in provider.stream([{"role": "user", "content": "Hi"}]):
            tokens.append(token)

        assert tokens == ["Hello", " world"]
        mock_stream.assert_called_once()
