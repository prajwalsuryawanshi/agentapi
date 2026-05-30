from agentapi.agent.tools import to_tool_definition, tool


def test_custom_tool_name_is_preserved_in_definition_and_schema():
    @tool(name="weather")
    def get_weather(city: str):
        pass

    definition = to_tool_definition(get_weather)

    assert definition.name == "weather"
    assert definition.schema["function"]["name"] == "weather"