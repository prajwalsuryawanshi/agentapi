"""Tool registration and JSON schema generation."""

from __future__ import annotations
import inspect
import json
import types
from dataclasses import dataclass
from typing import Any, get_args, get_origin
from collections.abc import Callable, Sequence
from agentapi.errors import AgentProviderError

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


def _unwrap_annotated(annotation: Any) -> Any:
    """Recursively unwrap Annotated annotations."""
    origin = get_origin(annotation)
    if origin is not None and str(origin).startswith("typing.Annotated"):
        args = get_args(annotation)
        if args:
            return _unwrap_annotated(args[0])
    return annotation


def has_explicit_default(param: inspect.Parameter) -> bool:
    """Check if the parameter has an explicit python default value."""
    return param.default is not inspect._empty


def resolve_parameter_default(param: inspect.Parameter) -> Any:
    """Resolve the default value of a parameter, if one exists."""
    return param.default


def _json_type(annotation: Any) -> str | list[str]:
    annotation = _unwrap_annotated(annotation)
    
    # Check if annotation is a subclass of BaseModel
    try:
        from pydantic import BaseModel
        if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
            return "object"
    except ImportError:
        pass

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


def _adjust_schema_for_strict(schema: dict[str, Any], strict: bool = True) -> dict[str, Any]:
    """Recursively adjust schema to comply with OpenAI strict mode requirements."""
    if not isinstance(schema, dict):
        return schema

    result = dict(schema)

    schema_type = result.get("type")
    if schema_type == "object" or "properties" in result:
        props = result.get("properties", {})
        if props:
            adjusted_props = {
                k: _adjust_schema_for_strict(v, strict=strict) for k, v in props.items()
            }
            result["properties"] = adjusted_props

            if strict:
                result["additionalProperties"] = False
                result["required"] = list(props.keys())

                for prop_name, prop_schema in adjusted_props.items():
                    if "default" in prop_schema:
                        ptype = prop_schema.get("type")
                        if ptype and not isinstance(ptype, list):
                            prop_schema["type"] = [ptype, "null"]
                        elif ptype and isinstance(ptype, list) and "null" not in ptype:
                            prop_schema["type"] = [*ptype, "null"]
        else:
            if strict:
                result["additionalProperties"] = False
                result["required"] = []

    if "items" in result:
        if isinstance(result["items"], dict):
            result["items"] = _adjust_schema_for_strict(result["items"], strict=strict)
        elif isinstance(result["items"], list):
            result["items"] = [_adjust_schema_for_strict(item, strict=strict) for item in result["items"]]

    return result


def _resolve_param_schema(annotation: Any, strict: bool = True) -> dict[str, Any]:
    """Resolve schema for a single parameter annotation."""
    annotation = _unwrap_annotated(annotation)

    # Check for Pydantic BaseModel subclass
    try:
        from pydantic import BaseModel
        if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
            schema = annotation.model_json_schema()
            return _adjust_schema_for_strict(schema, strict=strict)
    except ImportError:
        pass

    origin = get_origin(annotation)
    if origin is list or origin is Sequence:
        args = get_args(annotation)
        item_schema = {"type": "string"}
        if args:
            item_schema = _resolve_param_schema(args[0], strict=strict)
        return {
            "type": "array",
            "items": item_schema,
        }

    if origin is dict:
        return {
            "type": "object",
        }

    if origin in (types.UnionType, getattr(types, "UnionType", object)) or str(origin) == "typing.Union":
        args = get_args(annotation)
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1:
            base_schema = _resolve_param_schema(non_none[0], strict=strict)
            base_type = base_schema.get("type", "string")
            if isinstance(base_type, list):
                if "null" not in base_type:
                    base_schema["type"] = [*base_type, "null"]
            else:
                base_schema["type"] = [base_type, "null"]
            return base_schema
        elif len(non_none) > 1:
            return _resolve_param_schema(non_none[0], strict=strict)

    param_type = _json_type(annotation)
    return {"type": param_type}


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
    strict: bool = True,
) -> dict[str, Any]:
    signature = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in signature.parameters.items():
        annotation = param.annotation
        if annotation is inspect._empty:
            annotation = str

        param_schema = _resolve_param_schema(annotation, strict=strict)

        if strict:
            param_type = param_schema.get("type", "string")
            if param.default is not inspect._empty:
                if isinstance(param_type, list):
                    if "null" not in param_type:
                        param_schema["type"] = [*param_type, "null"]
                else:
                    param_schema["type"] = [param_type, "null"]

                try:
                    from pydantic import BaseModel
                    if isinstance(param.default, BaseModel):
                        param_schema["default"] = param.default.model_dump()
                    else:
                        param_schema["default"] = param.default
                except ImportError:
                    param_schema["default"] = param.default
            
            param_schema["description"] = f"Parameter: {param_name}"
            properties[param_name] = param_schema
            required.append(param_name)
        else:
            if param.default is not inspect._empty:
                try:
                    from pydantic import BaseModel
                    if isinstance(param.default, BaseModel):
                        param_schema["default"] = param.default.model_dump()
                    else:
                        param_schema["default"] = param.default
                except ImportError:
                    param_schema["default"] = param.default
            else:
                required.append(param_name)

            param_schema["description"] = f"Parameter: {param_name}"
            properties[param_name] = param_schema

    schema = {
        "type": "function",
        "function": {
            "name": (name or func.__name__).strip(),
            "description": _compose_tool_description(func, description=description, context=context),
            "parameters": {
                "type": "object",
                "properties": properties,
            },
        },
    }

    if strict:
        schema["function"]["strict"] = True
        schema["function"]["parameters"]["required"] = required
        schema["function"]["parameters"]["additionalProperties"] = False
    else:
        schema["function"]["strict"] = False
        schema["function"]["parameters"]["additionalProperties"] = True
        if required:
            schema["function"]["parameters"]["required"] = required

    return schema


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    context: str | None = None,
    strict: bool = True,
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
                strict=strict,
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
    try:
        return json.loads(args_json)
    except json.JSONDecodeError as exc:
        raise AgentProviderError(
            f"Failed to parse tool arguments as JSON: {exc}. Raw input: {args_json[:200]!r}",
            status_code=422,
        ) from exc
