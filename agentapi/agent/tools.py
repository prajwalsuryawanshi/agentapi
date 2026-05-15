"""Tool registration and JSON schema generation."""

from __future__ import annotations

import inspect
import json
import re
import types
from dataclasses import dataclass
from typing import Any, get_args, get_origin
from collections.abc import Callable

@dataclass
class ToolDefinition:
    """Internal representation of a callable tool."""

    name: str
    description: str
    context: str
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


def parse_docstring_params(func: Callable[..., Any]) -> dict[str, str]:
    """Parse parameter descriptions from a function's docstring.

    Supports Google style and NumPy style docstrings.
    Falls back to an empty dict if parsing fails or no docstring is present.

    Args:
        func: The callable whose docstring will be parsed.

    Returns:
        A dict mapping parameter names to their descriptions.

    Examples:
        Google style::

            def fn(city: str, units: str):
                \"\"\"Get weather.

                Args:
                    city (str): The city name to look up.
                    units (str): Temperature units, e.g. 'metric'.
                \"\"\"

        NumPy style::

            def fn(city: str, units: str):
                \"\"\"Get weather.

                Parameters
                ----------
                city : str
                    The city name to look up.
                units : str
                    Temperature units, e.g. 'metric'.
                \"\"\"
    """
    try:
        docstring = inspect.getdoc(func)
        if not docstring:
            return {}

        # Try Google style: Args:\n    param (type): description
        google_match = re.search(r"Args:\s*\n(.*?)(?:\n\s*\n|\Z)", docstring, re.DOTALL)
        if google_match:
            params: dict[str, str] = {}
            block = google_match.group(1)
            # Each param may span multiple continuation lines.
            # A new param starts at indent >= 2 followed by a word and colon.
            for match in re.finditer(
                r"^\s{2,}(\w+)(?:\s*\([^)]*\))?\s*:\s*(.+?)(?=\n\s{2,}\w+(?:\s*\([^)]*\))?\s*:|\Z)",
                block,
                re.MULTILINE | re.DOTALL,
            ):
                param_name = match.group(1).strip()
                param_desc = " ".join(match.group(2).split())
                params[param_name] = param_desc
            if params:
                return params

        # Try NumPy style: Parameters\n----------\nparam : type\n    description
        numpy_match = re.search(
            r"Parameters\s*\n[-]+\s*\n(.*?)(?:\n\s*\n[A-Za-z]|\Z)", docstring, re.DOTALL
        )
        if numpy_match:
            params = {}
            block = numpy_match.group(1)
            # Split on param entries: "name : type\n    description"
            for match in re.finditer(
                r"^(\w+)\s*(?::\s*[^\n]*)?\n((?:\s+.+\n?)+)",
                block,
                re.MULTILINE,
            ):
                param_name = match.group(1).strip()
                param_desc = " ".join(match.group(2).split())
                params[param_name] = param_desc
            if params:
                return params

        return {}

    except Exception:
        # Always fall back gracefully — never break tool registration
        return {}


def _compose_tool_description(
    func: Callable[..., Any],
    *,
    description: str | None = None,
    context: str | None = None,
) -> str:
    base_description = (description or inspect.getdoc(func) or f"Tool function {func.__name__}").strip()
    context_text = (context or getattr(func, "__agentapi_tool_context__", None) or "").strip()

    if context_text and context_text not in base_description:
        return f"{base_description}\n\nContext: {context_text}"

    return base_description


def _build_openai_tool_schema(
    func: Callable[..., Any],
    *,
    description: str | None = None,
    context: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    signature = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    # Parse docstring param descriptions once for all parameters
    docstring_params = parse_docstring_params(func)

    for param_name, param in signature.parameters.items():
        annotation = param.annotation
        if annotation is inspect._empty:
            annotation = str

        param_type = _json_type(annotation)
        if param.default is not inspect._empty and not isinstance(param_type, list):
            param_type = [param_type, "null"]

        # Use parsed docstring description if available, else fall back to generic
        param_description = docstring_params.get(param_name) or f"Parameter: {param_name}"

        properties[param_name] = {
            "type": param_type,
            "description": param_description,
        }

        # Strict mode expects required to include all declared properties.
        required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": (name or func.__name__).strip(),
            "description": _compose_tool_description(func, description=description, context=context),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    context: str | None = None,
) -> Callable[..., Any]:
    """Decorator that tags a Python function as an AgentAPI tool.

    The decorator supports explicit metadata so the caller can provide LLM-facing
    context at tool creation time without relying on docstrings alone.
    """

    def decorator(target: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = (name or target.__name__).strip()
        tool_description = _compose_tool_description(target, description=description, context=context)

        setattr(target, "__agentapi_tool_name__", tool_name)
        setattr(target, "__agentapi_tool_description__", tool_description)
        setattr(target, "__agentapi_tool_context__", (context or "").strip())
        setattr(
            target,
            "__agentapi_tool_schema__",
            _build_openai_tool_schema(
                target,
                description=description,
                context=context,
                name=tool_name,
            ),
        )
        return target

    if func is not None:
        return decorator(func)

    return decorator


def to_tool_definition(func: Callable[..., Any]) -> ToolDefinition:
    schema = getattr(func, "__agentapi_tool_schema__", None)
    if not schema:
        schema = _build_openai_tool_schema(func)

    description = getattr(func, "__agentapi_tool_description__", None) or inspect.getdoc(func) or ""
    context = getattr(func, "__agentapi_tool_context__", None) or ""

    return ToolDefinition(
        name=func.__name__,
        description=description,
        context=context,
        func=func,
        schema=schema,
    )


def parse_tool_args(args_json: str) -> dict[str, Any]:
    """Parse model tool arguments safely."""

    if not args_json.strip():
        return {}
    return json.loads(args_json)