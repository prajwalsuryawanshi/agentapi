"""Tool schema parameter description coverage."""

from __future__ import annotations

from agentapi import tool
from agentapi.agent.tools import to_tool_definition


def _properties(func):
    """Return generated JSON schema properties for a decorated tool."""
    definition = to_tool_definition(func)
    return definition.schema["function"]["parameters"]["properties"]


@tool
def google_weather(city: str, units: str = "metric") -> str:
    """Get current weather.

    Args:
        city (str): City name or "lat,lon" coordinates to query.
        units: Unit system to use for the forecast.
            Supports metric or imperial.
    """

    return f"{city} in {units}"


@tool
def numpy_search(query: str, limit: int) -> list[str]:
    """Search stored documents.

    Parameters
    ----------
    query : str
        Natural-language search query.
    limit : int
        Maximum number of results to return.

    Returns
    -------
    list[str]
        Matching document IDs.
    """

    return [query] * limit


@tool
def undocumented(count: int) -> int:
    # Deliberately no docstring: validates the generic fallback path.
    return count


@tool
def summary_only(topic: str) -> str:
    """Summarize a topic without parameter sections."""

    return topic


def test_google_style_docstring_param_descriptions_are_used() -> None:
    """Google-style Args sections should replace generic schema descriptions."""
    properties = _properties(google_weather)

    assert properties["city"]["description"] == 'City name or "lat,lon" coordinates to query.'
    assert properties["units"]["description"] == (
        "Unit system to use for the forecast. Supports metric or imperial."
    )


def test_numpy_style_docstring_param_descriptions_are_used() -> None:
    """NumPy Parameters sections should stop before the next docstring section."""
    properties = _properties(numpy_search)

    assert properties["query"]["description"] == "Natural-language search query."
    assert properties["limit"]["description"] == "Maximum number of results to return."


def test_missing_docstring_falls_back_to_generic_parameter_description() -> None:
    """Tools without docstrings should preserve existing fallback descriptions."""
    properties = _properties(undocumented)

    assert properties["count"]["description"] == "Parameter: count"


def test_docstring_without_param_section_falls_back_to_generic_description() -> None:
    """Docstrings without parameter sections should not alter parameter fallback text."""
    properties = _properties(summary_only)

    assert properties["topic"]["description"] == "Parameter: topic"
