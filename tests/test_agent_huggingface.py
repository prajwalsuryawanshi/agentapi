"""Test Agent integration with Hugging Face provider."""

from unittest.mock import AsyncMock, patch
import pytest
from agentapi import Agent
from agentapi.providers.huggingface import HuggingFaceProvider


@patch.dict("os.environ", {"HUGGINGFACE_API_KEY": "hf_test_token"})
def test_agent_huggingface_initialization():
    """Test that the agent initializes correctly with the huggingface provider."""
    agent = Agent(
        system_prompt="You are a helpful assistant",
        provider="huggingface",
    )
    assert agent.provider_name == "huggingface"
    assert agent.model == "Qwen/Qwen2.5-72B-Instruct"
    
    # Verify the instantiated provider is correct
    provider = agent._get_provider()
    assert isinstance(provider, HuggingFaceProvider)
    assert provider.api_key == "hf_test_token"
    assert provider.model == "Qwen/Qwen2.5-72B-Instruct"
    assert provider.base_url == "https://api-inference.huggingface.co/v1"


@patch.dict("os.environ", {
    "HUGGINGFACE_API_KEY": "hf_test_token",
    "HUGGINGFACE_BASE_URL": "https://custom-endpoint.huggingface.cloud/v1"
})
def test_agent_huggingface_custom_url_initialization():
    """Test that the agent initializes correctly with a custom base URL."""
    agent = Agent(
        system_prompt="You are a helpful assistant",
        provider="huggingface",
    )
    provider = agent._get_provider()
    assert isinstance(provider, HuggingFaceProvider)
    assert provider.api_key == "hf_test_token"
    assert provider.base_url == "https://custom-endpoint.huggingface.cloud/v1"
