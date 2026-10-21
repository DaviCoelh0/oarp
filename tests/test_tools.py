import pytest

from oarp.core.tools import InfrastructureError, Tool, ToolError, ToolRegistry, ToolSpec
from oarp.core.types import ErrorKind, ToolCall


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
    ("name", "args", "expected", "kind"),
    [
        ("nope", {"text": "x"}, "unknown tool 'nope'", ErrorKind.UNKNOWN_TOOL),
        ("echo", {}, "missing required argument(s): text", ErrorKind.INVALID_ARGUMENTS),
        ("echo", {"text": "x", "extra": 1}, "unknown argument(s): extra", ErrorKind.INVALID_ARGUMENTS),
        ("echo", {"text": 3}, "'text' must be of type string", ErrorKind.INVALID_ARGUMENTS),
        ("echo", {"text": "x", "times": True}, "'times' must be of type integer", ErrorKind.INVALID_ARGUMENTS),
        ("echo", "not a dict", "arguments must be a JSON object", ErrorKind.INVALID_ARGUMENTS),
        ("refuses", {"text": "x"}, "Error: not allowed", ErrorKind.TOOL_REFUSED),
        ("broken", {"text": "x"}, "tool raised KeyError", ErrorKind.TOOL_BUG),
    ],
)
def test_failures_become_error_results(
    registry: ToolRegistry, name: str, args: object, expected: str, kind: ErrorKind
) -> None:
    result = run(registry, name, args)
    assert result.is_error
    assert result.error_kind is kind
    assert expected in result.content


def test_successful_call_has_no_error_kind(registry: ToolRegistry) -> None:
    assert run(registry, "echo", {"text": "a"}).error_kind is None


def test_infrastructure_errors_are_re_raised_not_returned() -> None:
    def docker_down(text: str) -> str:
        raise InfrastructureError("docker daemon unavailable")

    registry = ToolRegistry([Tool(ToolSpec("needs_docker", "", SCHEMA), docker_down)])
    with pytest.raises(InfrastructureError, match="docker daemon"):
        registry.execute(ToolCall("c1", "needs_docker", {"text": "x"}))


def test_duplicate_names_rejected() -> None:
    tool = Tool(ToolSpec("echo", "", SCHEMA), _echo)
    with pytest.raises(ValueError, match="duplicate"):
        ToolRegistry([tool, tool])
