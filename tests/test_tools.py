import pytest

from oarp.core.tools import Tool, ToolError, ToolRegistry, ToolSpec
from oarp.core.types import ToolCall


def _echo(text: str, times: int = 1) -> str:
    return text * times


def _broken(text: str) -> str:
    raise KeyError("bug inside the tool")


def _refuses(text: str) -> str:
    raise ToolError("not allowed")


SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}, "times": {"type": "integer"}},
    "required": ["text"],
}


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(ToolSpec("echo", "Repeat text.", SCHEMA), _echo),
            Tool(ToolSpec("broken", "Always crashes.", SCHEMA), _broken),
            Tool(ToolSpec("refuses", "Always refuses.", SCHEMA), _refuses),
        ]
    )


def run(registry: ToolRegistry, name: str, args: object):
    return registry.execute(ToolCall("c1", name, args))  # type: ignore[arg-type]


def test_successful_call(registry: ToolRegistry) -> None:
    result = run(registry, "echo", {"text": "ab", "times": 2})
    assert (result.content, result.is_error, result.call_id) == ("abab", False, "c1")


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        ("nope", {"text": "x"}, "unknown tool 'nope'"),
        ("echo", {}, "missing required argument(s): text"),
        ("echo", {"text": "x", "extra": 1}, "unknown argument(s): extra"),
        ("echo", {"text": 3}, "'text' must be of type string"),
        ("echo", {"text": "x", "times": True}, "'times' must be of type integer"),
        ("echo", "not a dict", "arguments must be a JSON object"),
        ("refuses", {"text": "x"}, "Error: not allowed"),
        ("broken", {"text": "x"}, "tool raised KeyError"),
    ],
)
def test_failures_become_error_results(
    registry: ToolRegistry, name: str, args: object, expected: str
) -> None:
    result = run(registry, name, args)
    assert result.is_error
    assert expected in result.content


def test_duplicate_names_rejected() -> None:
    tool = Tool(ToolSpec("echo", "", SCHEMA), _echo)
    with pytest.raises(ValueError, match="duplicate"):
        ToolRegistry([tool, tool])
