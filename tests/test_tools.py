"""Tests for agentapi.agent.tools — parse_tool_args."""

import pytest
from agentapi.agent.tools import parse_tool_args
from agentapi.errors import AgentProviderError


def test_parse_tool_args_with_valid_json_string():
    assert parse_tool_args('{"location": "Tokyo"}') == {"location": "Tokyo"}

def test_parse_tool_args_with_empty_string():
    assert parse_tool_args("") == {}

def test_parse_tool_args_with_whitespace_string():
    assert parse_tool_args("   ") == {}

def test_parse_tool_args_with_none():
    assert parse_tool_args(None) == {}

def test_parse_tool_args_with_dict():
    assert parse_tool_args({"location": "Tokyo"}) == {"location": "Tokyo"}

def test_parse_tool_args_with_empty_dict():
    assert parse_tool_args({}) == {}

def test_parse_tool_args_with_invalid_json_raises():
    with pytest.raises(AgentProviderError):
        parse_tool_args("not valid json")

def test_parse_tool_args_with_unsupported_type_raises():
    with pytest.raises(TypeError):
        parse_tool_args(123)