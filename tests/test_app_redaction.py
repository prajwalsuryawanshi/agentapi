from fastapi.testclient import TestClient
from agentapi.core.app import AgentAPI
from agentapi.errors import AgentProviderError
import pytest


def _make_failing_app(error_redaction: bool = True, extra_patterns=None):
    kwargs = {"error_redaction": error_redaction}
    if extra_patterns is not None:
        kwargs["error_redaction_patterns"] = extra_patterns
    app = AgentAPI(**kwargs)

    @app.chat("/chat")
    async def failing_chat(message: str):
        raise AgentProviderError(
            "Provider returned 401 for key sk-abcdefghij1234567890XYZ",
            status_code=401,
        )

    return TestClient(app)


def test_json_error_redacts_api_key():
    client = _make_failing_app()
    resp = client.post("/chat?message=hi")
    body = resp.json()
    assert "sk-" not in body["error"]
    assert "[REDACTED]" in body["error"]


def test_redaction_opt_out_returns_raw():
    client = _make_failing_app(error_redaction=False)
    resp = client.post("/chat?message=hi")
    body = resp.json()
    assert "sk-abcdefghij1234567890XYZ" in body["error"]


def test_custom_pattern_additive_to_defaults():
    app = AgentAPI(error_redaction_patterns=[r"COMPANY-[A-Z]+"])

    @app.chat("/chat")
    async def fails(message: str):
        raise AgentProviderError(
            "Failed with sk-abcdefghij1234567890XYZ and COMPANY-SECRET",
            status_code=500,
        )

    client = TestClient(app)
    resp = client.post("/chat?message=hi")
    body = resp.json()
    assert "sk-" not in body["error"]
    assert "COMPANY-" not in body["error"]
    
def test_unknown_exception_type_still_redacted():
    """Generic exceptions from provider SDKs must also be redacted —
    they're the ones most likely to leak credentials in error messages."""
    app = AgentAPI()

    async def streaming_generator():
        yield "ok"
        raise RuntimeError("upstream failed with key sk-abcdefghij1234567890XYZ")

    @app.chat("/stream")
    async def streaming_endpoint(message: str):
        return streaming_generator()

    client = TestClient(app)
    app._sse_heartbeat_seconds = 0.5
    
    with client.stream("POST", "/stream?message=hi") as resp:
        content = b"".join(resp.iter_bytes()).decode()
    
    assert "sk-" not in content
    assert "[REDACTED]" in content
    


class TestKwargValidation:
    def test_error_redaction_string_raises_type_error(self):
        """String 'false' is truthy; reject it loudly instead of silently keeping redaction on."""
        with pytest.raises(TypeError, match="must be a bool"):
            AgentAPI(error_redaction="false")

    def test_error_redaction_int_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a bool"):
            AgentAPI(error_redaction=0)

    def test_error_redaction_patterns_bare_string_raises(self):
        """A single regex string would get exploded into character patterns by list(); reject it."""
        with pytest.raises(TypeError, match="not a single string"):
            AgentAPI(error_redaction_patterns=r"sk-\w+")

    def test_error_redaction_patterns_non_iterable_raises(self):
        with pytest.raises(TypeError, match="must be a list or tuple"):
            AgentAPI(error_redaction_patterns=42)

    def test_error_redaction_patterns_non_string_item_raises(self):
        with pytest.raises(TypeError, match="must contain only strings"):
            AgentAPI(error_redaction_patterns=[123, r"sk-\w+"])

    def test_error_redaction_patterns_tuple_accepted(self):
        """Tuples should work as iterables alongside lists."""
        app = AgentAPI(error_redaction_patterns=(r"CUSTOM-[A-Z]+",))
        assert any("CUSTOM" in p for p in app._error_redaction_patterns)