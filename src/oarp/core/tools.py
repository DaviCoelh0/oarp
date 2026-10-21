"""Tool protocol: what a tool looks like and how the harness runs one.

The registry is the single place where tool failures are caught, so a bad
call never crashes the run; it becomes an error result the model can read.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from oarp.core.types import ErrorKind, ToolCall, ToolResult, ToolSpec


class ToolError(Exception):
    """An expected, model-facing failure (bad path, file not found, ...)."""


class InfrastructureError(Exception):
    """The harness itself failed (e.g. Docker unavailable). Not the model's fault:
    the registry re-raises it, so the run aborts and can be marked invalid
    instead of being scored as an agent failure."""


@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    handler: Callable[..., str]  # called with the validated arguments as kwargs


_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


def validate_arguments(schema: dict[str, Any], arguments: Any) -> None:
    """Minimal check of flat JSON-Schema objects: required keys, no extra keys,
    primitive types. Raises ToolError with a message aimed at the model."""
    if not isinstance(arguments, dict):
        raise ToolError("arguments must be a JSON object")
    properties: dict[str, Any] = schema.get("properties", {})
    missing = [k for k in schema.get("required", []) if k not in arguments]
    if missing:
        raise ToolError(f"missing required argument(s): {', '.join(missing)}")
    unknown = [k for k in arguments if k not in properties]
    if unknown:
        raise ToolError(f"unknown argument(s): {', '.join(unknown)}")
    for key, value in arguments.items():
        expected = _JSON_TYPES.get(properties[key].get("type", ""))
        # bool is a subclass of int in Python; JSON treats them as distinct.
        wrong_bool = isinstance(value, bool) and properties[key].get("type") != "boolean"
        if expected is not None and (not isinstance(value, expected) or wrong_bool):
            raise ToolError(f"argument '{key}' must be of type {properties[key]['type']}")


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.spec.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.spec.name}")
        self._tools[tool.spec.name] = tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def execute(self, call: ToolCall) -> ToolResult:
        def error(kind: ErrorKind, message: str) -> ToolResult:
            return ToolResult(call.id, call.name, f"Error: {message}", is_error=True, error_kind=kind)

        tool = self._tools.get(call.name)
        if tool is None:
            return error(ErrorKind.UNKNOWN_TOOL, f"unknown tool '{call.name}'. Available: {', '.join(self._tools)}")
        # Separate try blocks: a ToolError from validation (bad arguments) and a
        # ToolError from the handler (the tool refused) are different kinds.
        try:
            validate_arguments(tool.spec.parameters, call.arguments)
        except ToolError as exc:
            return error(ErrorKind.INVALID_ARGUMENTS, str(exc))
        try:
            return ToolResult(call.id, call.name, tool.handler(**call.arguments))
        except ToolError as exc:
            return error(ErrorKind.TOOL_REFUSED, str(exc))
        except InfrastructureError:
            raise
        except Exception as exc:  # a tool bug must not crash the run
            return error(ErrorKind.TOOL_BUG, f"tool raised {type(exc).__name__}: {exc}")
