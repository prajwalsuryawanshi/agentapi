"""Tool registration and JSON schema generation."""

from __future__ import annotations

import inspect
import json
import types
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
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

    for param_name, param in signature.parameters.items():
        annotation = param.annotation

        if annotation is inspect._empty:
            annotation = str

        param_type = _json_type(annotation)

        if param.default is not inspect._empty and not isinstance(param_type, list):
            param_type = [param_type, "null"]

        properties[param_name] = {
            "type": param_type,
            "description": f"Parameter: {param_name}",
        }

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
        name=getattr(func, "__agentapi_tool_name__", func.__name__),
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


class BaseToolArgumentValidator(ABC):
    """Abstract base — subclass this to plug in a custom validator."""

    @abstractmethod
    def validate(
        self,
        func: Callable[..., Any],
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Validate and coerce args for func.

        Returns:
            dict with validated/coerced args on success.
            dict with key "error" on failure  →  {"error": "message"}
        """
        ...                                   


def _resolve_annotation(func, annotation):
    if not isinstance(annotation, str):
        return annotation
    try:
        globalns = getattr(func, "__globals__", {})
        return eval(annotation, globalns)    
    except Exception:
        return annotation


def _type_name(t):
    return getattr(t, "__name__", repr(t))



def _coerce_bool(v):
    if isinstance(v, bool):
        return v

    if isinstance(v, int):
        return bool(v)

    if isinstance(v, str):

        lowered = v.lower()

        if lowered in ("true", "1", "yes"):
            return True

        if lowered in ("false", "0", "no"):
            return False

    raise ValueError


def _coerce_list(v):

    if isinstance(v, str):

        result = json.loads(v)

        if not isinstance(result, list):
            raise ValueError

        return result

    raise ValueError


def _coerce_dict(v):

    if isinstance(v, str):

        result = json.loads(v)

        if not isinstance(result, dict):
            raise ValueError

        return result

    raise ValueError



COERCIBLE = {
    int: (str,),
    float: (str, int),
    bool: (str, int),
    str: (int, float, bool),
    list: (str,),
    dict: (str,),
    datetime: (str,),
    Path: (str,),
}


COERCE_FN = {
    int: int,
    float: float,
    bool: _coerce_bool,
    str: str,
    list: _coerce_list,
    dict: _coerce_dict,
    datetime: datetime.fromisoformat,
    Path: Path,
}


class ToolArgumentValidator(BaseToolArgumentValidator):

    def validate(
        self,
        func: Callable[..., Any],
        args: dict[str, Any]
    ) -> dict[str, Any]:

        sig = inspect.signature(func)

        errors = []

        for param_name, param in sig.parameters.items():

            if param_name == "self":
                continue

         
            if (
                param.default is inspect.Parameter.empty
                and param_name not in args
            ):

                errors.append(
                    f"missing required argument: {param_name}"
                )

                continue

            if param_name not in args:
                continue

            expected_type = _resolve_annotation(func, param.annotation) 
            value = args.get(param_name)

            if expected_type is inspect.Parameter.empty:
                continue

            try:

                if (
                    isinstance(value, expected_type)
                    and not (
                        expected_type is int
                        and type(value) is bool
                    )
                ):
                    continue

            except TypeError:
                pass

            if (
                expected_type in COERCIBLE
                and isinstance(
                    value,
                    COERCIBLE[expected_type]
                )
            ):

                try:

                    args[param_name] = (
                        COERCE_FN[expected_type](value)
                    )

                except (ValueError, TypeError):
                    errors.append(
                        f"invalid {_type_name(expected_type)} "    
                        f"for argument: {param_name}"
                    )

            else:
                errors.append(
                    f"argument '{param_name}': expected "
                    f"{_type_name(expected_type)}, "               
                    f"got {type(value).__name__}"
                )

        if errors:
            return {"error": errors[0]}

        return args