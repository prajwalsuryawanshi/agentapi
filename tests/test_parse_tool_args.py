import pytest

from agentapi.agent.tools import parse_tool_args
from agentapi.errors import AgentProviderError


def test_parse_tool_args_accepts_preparsed_mapping():
    args = parse_tool_args({"location": "Tokyo", "units": "metric"})

    assert args == {"location": "Tokyo", "units": "metric"}


def test_parse_tool_args_treats_none_as_empty_args():
    assert parse_tool_args(None) == {}


def test_parse_tool_args_rejects_unsupported_non_string_inputs():
    with pytest.raises(AgentProviderError, match="list"):
        parse_tool_args(["not", "a", "mapping"])


@pytest.mark.parametrize("args_json, expected_type", [
    ("[]", "list"),
    ("123", "int"),
    ("null", "NoneType"),
])
def test_parse_tool_args_rejects_json_strings_that_decode_to_non_objects(args_json, expected_type):
    with pytest.raises(AgentProviderError, match=expected_type):
        parse_tool_args(args_json)
