"""Unit tests for parse_docstring_params() in agentapi/agent/tools.py"""

import pytest
from agentapi.agent.tools import parse_docstring_params, _build_openai_tool_schema, tool


# Fixtures

def google_style_func(city: str, units: str) -> str:
    """Get the current weather for a city.

    Args:
        city (str): The city name to look up.
        units (str): Temperature units, either 'metric' or 'imperial'.
    """
    return f"{city} {units}"


def numpy_style_func(city: str, units: str) -> str:
    """Get the current weather for a city.

    Parameters
    ----------
    city : str
        The city name to look up.
    units : str
        Temperature units, either 'metric' or 'imperial'.
    """
    return f"{city} {units}"


def no_docstring_func(city: str, units: str) -> str:
    return f"{city} {units}"


def no_args_section_func(city: str, units: str) -> str:
    """Get the current weather for a city. No parameter docs here."""
    return f"{city} {units}"


def partial_args_func(city: str, units: str) -> str:
    """Get weather.

    Args:
        city (str): The city name to look up.
    """
    return f"{city} {units}"

# parse_docstring_params tests

class TestParseDocstringParams:

    def test_google_style_parses_all_params(self):
        result = parse_docstring_params(google_style_func)
        assert result["city"] == "The city name to look up."
        assert result["units"] == "Temperature units, either 'metric' or 'imperial'."

    def test_numpy_style_parses_all_params(self):
        result = parse_docstring_params(numpy_style_func)
        assert result["city"] == "The city name to look up."
        assert result["units"] == "Temperature units, either 'metric' or 'imperial'."

    def test_no_docstring_returns_empty_dict(self):
        result = parse_docstring_params(no_docstring_func)
        assert result == {}

    def test_no_args_section_returns_empty_dict(self):
        result = parse_docstring_params(no_args_section_func)
        assert result == {}

    def test_partial_args_returns_only_documented_params(self):
        result = parse_docstring_params(partial_args_func)
        assert "city" in result
        assert "units" not in result

    def test_returns_dict_type(self):
        result = parse_docstring_params(google_style_func)
        assert isinstance(result, dict)


# Schema injection tests

class TestSchemaInjection:

    def test_google_style_description_injected_into_schema(self):
        schema = _build_openai_tool_schema(google_style_func)
        props = schema["function"]["parameters"]["properties"]
        assert props["city"]["description"] == "The city name to look up."
        assert props["units"]["description"] == "Temperature units, either 'metric' or 'imperial'."

    def test_numpy_style_description_injected_into_schema(self):
        schema = _build_openai_tool_schema(numpy_style_func)
        props = schema["function"]["parameters"]["properties"]
        assert props["city"]["description"] == "The city name to look up."
        assert props["units"]["description"] == "Temperature units, either 'metric' or 'imperial'."

    def test_fallback_to_generic_when_no_docstring(self):
        schema = _build_openai_tool_schema(no_docstring_func)
        props = schema["function"]["parameters"]["properties"]
        assert props["city"]["description"] == "Parameter: city"
        assert props["units"]["description"] == "Parameter: units"

    def test_fallback_to_generic_when_no_args_section(self):
        schema = _build_openai_tool_schema(no_args_section_func)
        props = schema["function"]["parameters"]["properties"]
        assert props["city"]["description"] == "Parameter: city"
        assert props["units"]["description"] == "Parameter: units"

    def test_partial_docs_mixed_fallback(self):
        schema = _build_openai_tool_schema(partial_args_func)
        props = schema["function"]["parameters"]["properties"]
        assert props["city"]["description"] == "The city name to look up."
        assert props["units"]["description"] == "Parameter: units"  # fallback


#@tool decorator integration tests

class TestToolDecoratorIntegration:

    def test_tool_decorator_uses_docstring_descriptions(self):
        @tool
        def get_weather(city: str, units: str) -> str:
            """Fetch weather data.

            Args:
                city (str): The city to fetch weather for.
                units (str): The unit system to use.
            """
            return f"{city} {units}"

        schema = get_weather.__agentapi_tool_schema__
        props = schema["function"]["parameters"]["properties"]
        assert props["city"]["description"] == "The city to fetch weather for."
        assert props["units"]["description"] == "The unit system to use."

    def test_tool_decorator_fallback_without_docstring(self):
        @tool
        def simple_tool(x: int, y: int) -> int:
            return x + y

        schema = simple_tool.__agentapi_tool_schema__
        props = schema["function"]["parameters"]["properties"]
        assert props["x"]["description"] == "Parameter: x"
        assert props["y"]["description"] == "Parameter: y"