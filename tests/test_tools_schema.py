from agentapi.agent.tools import _build_openai_tool_schema


def test_optional_param_not_marked_required():
    def sample(a: int, b: str = "default"):
        pass

    schema = _build_openai_tool_schema(sample)
    params = schema["function"]["parameters"]
    required = params.get("required", [])

    assert "a" in required
    assert "b" not in required
