import pytest

from agentapi.agent.tools import parse_tool_args
from agentapi.errors import AgentProviderError


def test_parse_tool_args_accepts_json_object_string() -> None:
    assert parse_tool_args('{"order_id": "A-42", "retries": 2}') == {
        "order_id": "A-42",
        "retries": 2,
    }


def test_parse_tool_args_accepts_provider_native_mapping() -> None:
    raw_args = {"city": "Delhi", "limit": 3}

    parsed = parse_tool_args(raw_args)

    assert parsed == raw_args
    assert parsed is not raw_args


def test_parse_tool_args_treats_missing_args_as_empty() -> None:
    assert parse_tool_args(None) == {}
    assert parse_tool_args("") == {}
    assert parse_tool_args("   ") == {}


@pytest.mark.parametrize("raw_args", [["not", "an", "object"], 42, True])
def test_parse_tool_args_rejects_non_string_non_mapping_inputs(raw_args: object) -> None:
    with pytest.raises(AgentProviderError) as exc_info:
        parse_tool_args(raw_args)

    assert exc_info.value.status_code == 422
    assert "JSON object string or mapping" in str(exc_info.value)


@pytest.mark.parametrize("raw_args", ['["not", "an", "object"]', '"text"', "42"])
def test_parse_tool_args_rejects_json_values_that_are_not_objects(raw_args: str) -> None:
    with pytest.raises(AgentProviderError) as exc_info:
        parse_tool_args(raw_args)

    assert exc_info.value.status_code == 422
    assert "decode to a JSON object" in str(exc_info.value)
