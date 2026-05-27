"""Error response redaction tests for AgentAPI routes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agentapi import AgentAPI, AgentProviderError


DEFAULT_SECRETS = (
    "sk-testsecret123456",
    "gsk_testsecret123456",
    "AIzaTestSecret123456",
    "Bearer abcdefghijklmnopqrstuvwxyz",
    "https://user:password@example.com/v1/chat",
)


@pytest.mark.parametrize("secret", DEFAULT_SECRETS)
def test_chat_error_response_redacts_default_secret_patterns(secret: str) -> None:
    app = AgentAPI()

    @app.chat("/chat")
    async def chat() -> str:
        raise AgentProviderError(f"provider failed with {secret}", status_code=502)

    response = TestClient(app).post("/chat")

    assert response.status_code == 502
    assert secret not in response.json()["error"]
    assert "[REDACTED]" in response.json()["error"]


def test_sse_error_event_redacts_default_secret_patterns() -> None:
    app = AgentAPI()
    secret = "sk-streamsecret123456"

    async def stream_with_error():
        raise AgentProviderError(f"stream failed with {secret}", status_code=502)
        yield "unreachable"

    @app.stream("/stream")
    async def stream():
        return stream_with_error()

    response = TestClient(app).post("/stream")

    assert response.status_code == 200
    assert "event: error" in response.text
    assert secret not in response.text
    assert "[REDACTED]" in response.text


def test_heartbeat_sse_error_event_redacts_unexpected_errors() -> None:
    app = AgentAPI(sse_heartbeat_seconds=0.01)
    secret = "Bearer heartbeatsecret123456"

    async def stream_with_error():
        raise RuntimeError(f"unexpected provider failure with {secret}")
        yield "unreachable"

    @app.stream("/stream")
    async def stream():
        return stream_with_error()

    response = TestClient(app).post("/stream")

    assert response.status_code == 200
    assert "event: error" in response.text
    assert secret not in response.text
    assert "[REDACTED]" in response.text


def test_custom_error_redaction_patterns_are_additive() -> None:
    app = AgentAPI(error_redaction_patterns=[r"internal-ticket-\d+"])
    default_secret = "sk-customsecret123456"
    custom_secret = "internal-ticket-12345"

    @app.chat("/chat")
    async def chat() -> str:
        raise AgentProviderError(
            f"provider failed with {default_secret} and {custom_secret}"
        )

    response = TestClient(app).post("/chat")
    error = response.json()["error"]

    assert default_secret not in error
    assert custom_secret not in error
    assert error.count("[REDACTED]") == 2


def test_error_redaction_can_be_disabled() -> None:
    app = AgentAPI(error_redaction=False)
    secret = "sk-visiblewhenoptedout123456"

    @app.chat("/chat")
    async def chat() -> str:
        raise AgentProviderError(f"provider failed with {secret}")

    response = TestClient(app).post("/chat")

    assert response.status_code == 502
    assert secret in response.json()["error"]
