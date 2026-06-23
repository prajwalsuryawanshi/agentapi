import pytest
import asyncio
from agentapi.agent.agent import Agent
from agentapi.agent.tools import tool, to_tool_definition, ToolDefinition
from agentapi.providers.base import ToolCall

def test_tool_definition_custom_name_and_context():
    @tool(name="custom_math_tool", context="Calculates the sum of two integers")
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    t_def = to_tool_definition(add_numbers)
    assert isinstance(t_def, ToolDefinition)
    assert t_def.name == "custom_math_tool"
    assert t_def.context == "Calculates the sum of two integers"
    assert t_def.schema["function"]["name"] == "custom_math_tool"
    assert "Calculates the sum of two integers" in t_def.schema["function"]["description"]

    # Verify registration and execution in Agent
    agent = Agent(system_prompt="test", provider="openai")
    agent.add_tool(add_numbers)

    assert "custom_math_tool" in agent._tools
    assert "add_numbers" not in agent._tools

    # Simulate execution of tool call
    calls = [
        ToolCall(id="call_999", name="custom_math_tool", arguments='{"a": 5, "b": 10}')
    ]
    messages = []
    
    asyncio.run(agent._execute_tool_calls(calls, messages))
    
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["content"] == "15"
