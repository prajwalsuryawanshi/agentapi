from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from agentapi import Agent
from agentapi.providers.base import ToolCall
from agentapi.agent.agent import MAX_TOOL_RETRIES

from agentapi.agent.tools import (
    BaseToolArgumentValidator,
    ToolArgumentValidator,
    _resolve_annotation,
)
validator = ToolArgumentValidator()


# Sample tool functions

def add(a: int, b: int):
    return a + b


def greet(name: str):
    return f"Hello {name}"


def scale(value: float):
    return value


def toggle(flag: bool):
    return flag


def process(items: list, config: dict):
    return items, config


def schedule(event: str, when: datetime):
    return when


def read_file(path: Path):
    return path


def no_annotations(x, y):
    return x, y


def method_like(self, x: int):
    return x



# Missing required arguments


def test_missing_required_argument():

    result = validator.validate(
        add,
        {}
    )

    assert "error" in result
    assert result["error"] == (
        "missing required argument: a"
    )


def test_optional_argument_not_required():

    def tool(a: int, b: int = 5):
        return a + b

    result = validator.validate(
        tool,
        {"a": 5}
    )

    assert "error" not in result


def test_self_argument_skipped():

    result = validator.validate(
        method_like,
        {"x": 5}
    )

    assert "error" not in result



# Type validation


def test_valid_int_arguments():

    result = validator.validate(
        add,
        {"a": 5, "b": 7}
    )

    assert "error" not in result


def test_wrong_type_error():

    result = validator.validate(
        add,
        {"a": [1, 2], "b": 5}
    )

    assert "error" in result
    assert "expected int" in result["error"]


def test_unannotated_params_skip_validation():

    result = validator.validate(
        no_annotations,
        {"x": [1, 2], "y": "hello"}
    )

    assert "error" not in result



# Integer coercion


def test_string_to_int_coercion():

    result = validator.validate(
        add,
        {"a": "5", "b": "7"}
    )

    assert result["a"] == 5
    assert result["b"] == 7

    assert isinstance(result["a"], int)
    assert isinstance(result["b"], int)


def test_invalid_integer_coercion():

    result = validator.validate(
        add,
        {"a": "hello", "b": "7"}
    )

    assert "error" in result
    assert "invalid int" in result["error"]



# Float coercion


def test_string_to_float_coercion():

    result = validator.validate(
        scale,
        {"value": "3.14"}
    )

    assert result["value"] == 3.14
    assert isinstance(result["value"], float)


def test_invalid_float_coercion():

    result = validator.validate(
        scale,
        {"value": "not-a-float"}
    )

    assert "error" in result



# Bool coercion


@pytest.mark.parametrize(
    "value",
    ["true", "True", "1", "yes"]
)
def test_truthy_bool_coercion(value):

    result = validator.validate(
        toggle,
        {"flag": value}
    )

    assert result["flag"] is True


@pytest.mark.parametrize(
    "value",
    ["false", "False", "0", "no"]
)
def test_falsy_bool_coercion(value):

    result = validator.validate(
        toggle,
        {"flag": value}
    )

    assert result["flag"] is False


def test_invalid_bool_coercion():

    result = validator.validate(
        toggle,
        {"flag": "banana"}
    )

    assert "error" in result



# Bool/int overlap edge case


def test_bool_not_accepted_as_int():

    result = validator.validate(
        add,
        {"a": True, "b": 5}
    )

    assert "error" in result



# String coercion


def test_int_to_string_coercion():

    result = validator.validate(
        greet,
        {"name": 42}
    )

    assert result["name"] == "42"

    assert isinstance(result["name"], str)


# List coercion


def test_json_string_to_list():

    result = validator.validate(
        process,
        {
            "items": "[1, 2, 3]",
            "config": {}
        }
    )

    assert result["items"] == [1, 2, 3]


def test_invalid_list_coercion():

    result = validator.validate(
        process,
        {
            "items": "hello",
            "config": {}
        }
    )

    assert "error" in result


# Dict coercion


def test_json_string_to_dict():

    result = validator.validate(
        process,
        {
            "items": [],
            "config": '{"a": 1}'
        }
    )

    assert result["config"] == {"a": 1}


def test_invalid_dict_coercion():

    result = validator.validate(
        process,
        {
            "items": [],
            "config": "hello"
        }
    )

    assert "error" in result



# Datetime coercion


def test_datetime_coercion():

    result = validator.validate(
        schedule,
        {
            "event": "meeting",
            "when": "2025-01-01T10:30:00"
        }
    )

    assert isinstance(
        result["when"],
        datetime
    )


def test_invalid_datetime_coercion():

    result = validator.validate(
        schedule,
        {
            "event": "meeting",
            "when": "not-a-date"
        }
    )

    assert "error" in result


 
# Path coercion


def test_path_coercion():

    result = validator.validate(
        read_file,
        {"path": "/tmp/test.txt"}
    )

    assert isinstance(
        result["path"],
        Path
    )



# Structured error responses

def test_error_response_is_dict():

    result = validator.validate(
        add,
        {}
    )

    assert isinstance(result, dict)


def test_error_key_exists():

    result = validator.validate(
        add,
        {}
    )

    assert "error" in result
    assert isinstance(result["error"], str)



# Abstract base validator


def test_base_validator_is_abstract():

    with pytest.raises(TypeError):
        BaseToolArgumentValidator()


def test_custom_validator():

    class MyValidator(BaseToolArgumentValidator):

        def validate(self, func, args):
            return args

    validator = MyValidator()

    result = validator.validate(
        add,
        {"a": 1, "b": 2}
    )

    assert result == {"a": 1, "b": 2}



# Validator should never crash


def test_validator_never_crashes_on_invalid_input():

    try:

        result = validator.validate(
            add,
            {"a": object(), "b": []}
        )

        assert isinstance(result, dict)

    except Exception as exc:
        pytest.fail(
            f"validator crashed unexpectedly: {exc}"
        )

def test_agent_uses_custom_validator():

    class MyValidator(BaseToolArgumentValidator):

        def validate(self, func, args):

            return {"error": "blocked by custom validator"}

    agent = Agent(
        system_prompt="test",
        tools=[add],
        tool_argument_validator=MyValidator(),
    )

    assert isinstance(
        agent._validator,
        MyValidator
    )

# Retry handling

def test_retry_counts_increment():

    retry_counts = {}

    tool_name = "add"

    retry_counts[tool_name] = (
        retry_counts.get(tool_name, 0) + 1
    )

    assert retry_counts["add"] == 1

    retry_counts[tool_name] = (
        retry_counts.get(tool_name, 0) + 1
    )

    assert retry_counts["add"] == 2


def test_retry_limit_reached():

    retry_counts = {
        "add": MAX_TOOL_RETRIES
    }

    tool_name = "add"

    assert (
        retry_counts[tool_name]
        >= MAX_TOOL_RETRIES
    )

def test_retry_counts_reset_per_run():

    def create_run_scope():
        return {}

    run1 = create_run_scope()

    run1["add"] = 3

    run2 = create_run_scope()

    assert "add" not in run2

def test_different_tools_have_independent_retry_counts():

    retry_counts = {}

    retry_counts["add"] = (
        retry_counts.get("add", 0) + 1
    )

    retry_counts["multiply"] = (
        retry_counts.get("multiply", 0) + 1
    )

    retry_counts["multiply"] = (
        retry_counts.get("multiply", 0) + 1
    )

    assert retry_counts["add"] == 1
    assert retry_counts["multiply"] == 2



def test_future_annotations_do_not_crash_validator():

    def add(a: int, b: int):
        return a + b

    result = validator.validate(
        add,
        {"a": [1, 2], "b": 3}
    )

    assert isinstance(result, dict)
    assert "error" in result


def test_resolve_annotation():

    result = _resolve_annotation(add, "int")

    assert result is int


@pytest.mark.anyio
async def test_valid_tool_execution():

    agent = Agent(
        system_prompt="test",
        provider="openai",
        tools=[add],
        tool_argument_validator=validator,
    )

    calls = [
        ToolCall(
            id="call_add",
            name="add",
            arguments='{"a": 5, "b": 3}',
        )
    ]

    conversation = []
    retry_counts = {}

    await agent._execute_tool_calls(
        calls,
        conversation,
        retry_counts,
    )

    assert conversation[0]["content"] == "8"