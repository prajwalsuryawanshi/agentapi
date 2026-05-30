from agentapi.agent.tools import _build_openai_tool_schema, to_tool_definition, tool


def test_optional_param_not_marked_required():
    def sample(a: int, b: str = "default"):
        pass

    schema = _build_openai_tool_schema(sample)
    params = schema["function"]["parameters"]
    required = params.get("required", [])

    assert "a" in required
    assert "b" not in required


def test_custom_tool_name_is_preserved_in_definition_and_schema():
    @tool(name="weather")
    def get_weather(city: str):
        pass

    definition = to_tool_definition(get_weather)

    assert definition.name == "weather"
    assert definition.schema["function"]["name"] == "weather"
