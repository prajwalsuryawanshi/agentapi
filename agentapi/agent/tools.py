"""Tool registration and JSON schema generation."""

from __future__ import annotations

import inspect
import json
import types
from dataclasses import dataclass
from typing import Any, Callable, get_args, get_origin


@dataclass
class ToolDefinition:
    """Internal representation of a callable tool."""

    name: str
    description: str
    func: Callable[..., Any]
    schema: dict[str, Any]


_TYPE_MAP: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    dict: "object",
    list: "array",
}


def _json_type(annotation: Any) -> str | list[str]:
    origin = get_origin(annotation)
    if origin is None:
        return _TYPE_MAP.get(annotation, "string")

    if origin is list:
        return "array"
    if origin is dict:
        return "object"

    # Handle Optional[T] / Union[T, None] as nullable schema types.
    if origin in (types.UnionType, getattr(types, "UnionType", object)) or str(origin) == "typing.Union":
        args = get_args(annotation)
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1 and len(args) != 1:
            base_type = _json_type(non_none[0])
            if isinstance(base_type, list):
                return base_type if "null" in base_type else [*base_type, "null"]
            return [base_type, "null"]

    args = get_args(annotation)
    if origin is type(None) and not args:
        return "string"

    # Handle Optional[T] or Union[T, None].
    non_none = [arg for arg in args if arg is not type(None)]
    if non_none:
        return _json_type(non_none[0])

    return "string"


def _build_openai_tool_schema(func: Callable[..., Any]) -> dict[str, Any]:
    signature = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in signature.parameters.items():
        annotation = param.annotation
        if annotation is inspect._empty:
            annotation = str

        param_type = _json_type(annotation)
        if param.default is not inspect._empty and not isinstance(param_type, list):
            param_type = [param_type, "null"]

        properties[name] = {
            "type": param_type,
            "description": f"Parameter: {name}",
        }

        # Strict mode expects required to include all declared properties.
        required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": inspect.getdoc(func) or f"Tool function {func.__name__}",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def tool(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that tags a Python function as an AgentAPI tool."""

    setattr(func, "__agentapi_tool_schema__", _build_openai_tool_schema(func))
    return func


def to_tool_definition(func: Callable[..., Any]) -> ToolDefinition:
    schema = getattr(func, "__agentapi_tool_schema__", None)
    if not schema:
        schema = _build_openai_tool_schema(func)

    return ToolDefinition(
        name=func.__name__,
        description=inspect.getdoc(func) or "",
        func=func,
        schema=schema,
    )


def parse_tool_args(args_json: str) -> dict[str, Any]:
    """Parse model tool arguments safely."""

    if not args_json.strip():
        return {}
    return json.loads(args_json)
