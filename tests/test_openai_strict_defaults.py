import pytest
from typing import Annotated, Any
from pydantic import BaseModel
from agentapi.agent.agent import Agent
from agentapi.agent.tools import tool, to_tool_definition
from agentapi.providers.base import ToolCall


# 1. Function with primitive defaults
@tool(strict=True)
def tool_primitive_defaults(limit: int = 10, enabled: bool = True):
    return f"limit={limit}, enabled={enabled}"


@tool(strict=False)
def tool_primitive_defaults_non_strict(limit: int = 10, enabled: bool = True):
    return f"limit={limit}, enabled={enabled}"


def test_primitive_defaults_schema_and_execution():
    # Verify strict schema
    t_def = to_tool_definition(tool_primitive_defaults)
    schema = t_def.schema
    
    assert schema["function"]["strict"] is True
    params = schema["function"]["parameters"]["properties"]
    
    # Check types are nullable in strict schema
    assert "null" in params["limit"]["type"]
    assert "null" in params["enabled"]["type"]
    
    # Check default values are preserved in strict schema
    assert params["limit"]["default"] == 10
    assert params["enabled"]["default"] is True
    
    # Verify required array contains both parameters under strict mode
    assert "limit" in schema["function"]["parameters"]["required"]
    assert "enabled" in schema["function"]["parameters"]["required"]

    # Verify non-strict schema
    t_def_ns = to_tool_definition(tool_primitive_defaults_non_strict)
    schema_ns = t_def_ns.schema
    
    assert schema_ns["function"]["strict"] is False
    params_ns = schema_ns["function"]["parameters"]["properties"]
    
    # Check types are NOT forced nullable in non-strict schema
    assert params_ns["limit"]["type"] == "integer"
    assert params_ns["enabled"]["type"] == "boolean"
    assert params_ns["limit"]["default"] == 10
    assert params_ns["enabled"]["default"] is True
    
    # In non-strict mode, parameters with defaults are NOT required
    assert "required" not in schema_ns["function"]["parameters"]

    # Test Execution
    agent = Agent(system_prompt="test", provider="openai", model="gpt-4o-mini")
    agent.add_tool(tool_primitive_defaults)

    # Simulate LLM returning null/None under strict mode
    calls = [
        ToolCall(id="call_1", name="tool_primitive_defaults", arguments='{"limit": null, "enabled": null}')
    ]
    messages = []
    
    # Run tool execution (we must use an event loop since _execute_tool_calls is async)
    import asyncio
    asyncio.run(agent._execute_tool_calls(calls, messages))
    
    # Verify that default values were successfully restored
    assert len(messages) == 1
    assert messages[0]["content"] == "limit=10, enabled=True"


# 2. Explicit None default
@tool(strict=True)
def tool_explicit_none_default(name: str | None = None):
    return f"name={name}"


def test_explicit_none_default():
    t_def = to_tool_definition(tool_explicit_none_default)
    schema = t_def.schema
    
    # Check schema has default None
    params = schema["function"]["parameters"]["properties"]
    assert params["name"]["default"] is None
    assert "null" in params["name"]["type"]
    
    # Test Execution
    agent = Agent(system_prompt="test", provider="openai", model="gpt-4o-mini")
    agent.add_tool(tool_explicit_none_default)

    calls = [
        ToolCall(id="call_2", name="tool_explicit_none_default", arguments='{"name": null}')
    ]
    messages = []
    
    import asyncio
    asyncio.run(agent._execute_tool_calls(calls, messages))
    
    assert len(messages) == 1
    assert messages[0]["content"] == "name=None"


# 3. Required argument
@tool(strict=True)
def tool_required_arg(user_id: int):
    return f"user_id={user_id}"


def test_required_arg():
    t_def = to_tool_definition(tool_required_arg)
    schema = t_def.schema
    
    params = schema["function"]["parameters"]["properties"]
    # Required argument should have no default key
    assert "default" not in params["user_id"]
    
    # Required argument must be in required list
    assert "user_id" in schema["function"]["parameters"]["required"]


# 4. Mixed defaults + required args
@tool(strict=True)
def tool_mixed_args(user_id: int, limit: int = 5):
    return f"user_id={user_id}, limit={limit}"


@tool(strict=False)
def tool_mixed_args_ns(user_id: int, limit: int = 5):
    return f"user_id={user_id}, limit={limit}"


def test_mixed_args():
    # Strict
    t_def = to_tool_definition(tool_mixed_args)
    schema = t_def.schema
    assert "user_id" in schema["function"]["parameters"]["required"]
    assert "limit" in schema["function"]["parameters"]["required"]
    
    # Non-strict
    t_def_ns = to_tool_definition(tool_mixed_args_ns)
    schema_ns = t_def_ns.schema
    assert "user_id" in schema_ns["function"]["parameters"]["required"]
    assert "limit" not in schema_ns["function"]["parameters"]["required"]

    # Test Execution
    agent = Agent(system_prompt="test", provider="openai", model="gpt-4o-mini")
    agent.add_tool(tool_mixed_args)

    calls = [
        ToolCall(id="call_3", name="tool_mixed_args", arguments='{"user_id": 42, "limit": null}')
    ]
    messages = []
    
    import asyncio
    asyncio.run(agent._execute_tool_calls(calls, messages))
    
    assert len(messages) == 1
    assert messages[0]["content"] == "user_id=42, limit=5"


# 5. Regression case with Annotated and Pydantic models
class ConfigModel(BaseModel):
    timeout: int = 30
    retry: bool = True


@tool(strict=True)
def tool_complex_types(
    config: ConfigModel,
    tags: list[str],
    priority: Annotated[int, "priority level"] = 1,
):
    return f"timeout={config.timeout}, retry={config.retry}, tags={tags}, priority={priority}"


def test_complex_regression():
    t_def = to_tool_definition(tool_complex_types)
    schema = t_def.schema
    
    # Verify ConfigModel fields are recursively forced in strict schema
    properties = schema["function"]["parameters"]["properties"]
    assert "config" in properties
    config_schema = properties["config"]
    assert config_schema["type"] == "object"
    assert config_schema["additionalProperties"] is False
    assert "timeout" in config_schema["required"]
    assert "retry" in config_schema["required"]
    assert "null" in config_schema["properties"]["timeout"]["type"]
    assert "null" in config_schema["properties"]["retry"]["type"]
    
    # Verify list type works
    assert properties["tags"]["type"] == "array"
    
    # Test Execution
    agent = Agent(system_prompt="test", provider="openai", model="gpt-4o-mini")
    agent.add_tool(tool_complex_types)

    # Simulate LLM sending nulls inside nested Pydantic model and for priority default
    calls = [
        ToolCall(
            id="call_4",
            name="tool_complex_types",
            arguments='{"config": {"timeout": null, "retry": null}, "tags": ["prod"], "priority": null}'
        )
    ]
    messages = []
    
    import asyncio
    asyncio.run(agent._execute_tool_calls(calls, messages))
    
    # Verify defaults are recursively resolved correctly
    assert len(messages) == 1
    assert messages[0]["content"] == "timeout=30, retry=True, tags=['prod'], priority=1"


class ItemModel(BaseModel):
    name: str
    price: float = 9.99


@tool(strict=True)
def tool_wrapped_types(
    items: list[ItemModel],
    config_opt: ConfigModel | None = None,
    item_map: dict[str, ItemModel] | None = None,
):
    items_str = ", ".join(f"{item.name}:{item.price}" for item in items)
    config_str = f"timeout={config_opt.timeout}" if config_opt else "None"
    map_str = ", ".join(f"{k}->{v.price}" for k, v in item_map.items()) if item_map else "None"
    return f"items=[{items_str}], config={config_str}, map={map_str}"


def test_wrapped_types_resolution():
    agent = Agent(system_prompt="test", provider="openai")
    agent.add_tool(tool_wrapped_types)

    calls = [
        ToolCall(
            id="call_5",
            name="tool_wrapped_types",
            arguments='{"items": [{"name": "apple", "price": null}], "config_opt": {"timeout": null, "retry": null}, "item_map": {"fav": {"name": "banana", "price": null}}}'
        )
    ]
    messages = []

    import asyncio
    asyncio.run(agent._execute_tool_calls(calls, messages))

    assert len(messages) == 1
    assert messages[0]["content"] == "items=[apple:9.99], config=timeout=30, map=fav->9.99"
