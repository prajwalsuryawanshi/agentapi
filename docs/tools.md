# Tool Calling

Tool calling lets the model invoke your Python functions during a conversation.

## Define a Tool

You can define tools with plain `@tool` or with explicit metadata.

### Option A: Docstring-based

```python
from agentapi import tool


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city in plain text."""
    return f"Weather in {city}: sunny"
```

### Option B: Explicit metadata (recommended)

```python
from agentapi import tool


@tool(
    name="get_weather",
    description="Get current weather conditions for a city.",
    context="Use this for weather intent. Ask for city when missing.",
)
def get_weather(city: str) -> str:
    return f"Weather in {city}: sunny"
```

Metadata helps the model choose tools more reliably in ambiguous prompts.

## Attach Tools to an Agent

```python
from agentapi import Agent
from tools import get_weather

agent = Agent(
    system_prompt="You are a weather assistant",
    provider="openai",
    tools=[get_weather],
)
```

## How Tool Execution Works

1. User message is sent to provider with tool schemas.
2. Provider may return tool calls.
3. Agent executes matching Python functions.
4. Tool outputs are appended to conversation memory.
5. Agent asks provider again for final response.

## Authorize Tool Calls

Model-selected tools can be influenced by user prompts and previous tool output.
For sensitive tools, add an `authorize_tool` callback that inspects every tool call
before the Python function runs.

```python
from agentapi import Agent, tool
from agentapi.providers.base import ToolCall


@tool
def delete_account(account_id: str) -> str:
    return f"deleted {account_id}"


def authorize(call: ToolCall) -> bool | str:
    if call.name == "delete_account":
        return "This tool requires an explicit admin approval step."
    return True


agent = Agent(
    system_prompt="You are a safe support assistant.",
    provider="openai",
    tools=[delete_account],
    authorize_tool=authorize,
)
```

Return `True` to allow execution, `False` to deny with a default message, or a
string to deny with that custom message. Denied calls are returned to the model as
tool results, so the normal tool loop can continue without executing unsafe code.

## Security Notes

- Treat tool calls as untrusted until your application authorizes them.
- Expose only the tools needed for the current route or user role.
- Use `authorize_tool` for side-effecting actions such as deletes, payments,
  outbound email, or privileged data access.
- Treat tool output as untrusted content before sending it back to the model.
- Validate parsed arguments inside the tool, especially IDs and tenant boundaries.

## Recommended Tool Authoring

- Add explicit `description` and `context` for model-facing intent.
- Use clear argument names and type hints.
- Keep outputs deterministic and concise.
- Return machine-friendly strings for downstream parsing when needed.
- Handle exceptions inside the tool and return actionable failure text.

## Tool Design Example

```python
@tool(
    description="Lookup current stock quantity for a SKU.",
    context="Use before confirming order availability.",
)
def get_inventory(sku: str) -> str:
    try:
        qty = inventory_service.lookup(sku)
        return f"sku={sku}; quantity={qty}"
    except Exception:
        return f"sku={sku}; error=inventory_lookup_failed"
```

## Parsing and Validation

AgentAPI safely parses tool arguments from model JSON payloads before execution.

## Architecture Reference

For schema generation, execution ordering, persistence behavior, and provider
translation requirements, see the [tool runtime flow](architecture/runtime.md#tool-registration-and-execution),
[tool object model](architecture/object-model.md#tool-objects-and-functions), and
[tool contribution guide](architecture/contributing.md#changing-tools-or-the-tool-loop).
