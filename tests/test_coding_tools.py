import pytest
from conftest import TASKS_DIR, requires_docker

from oarp.core.loop import run_agent
from oarp.core.model import ScriptedModel
from oarp.core.tools import ToolRegistry
from oarp.core.types import ModelResponse, StopReason, ToolCall
from oarp.envs.coding import sandbox
from oarp.envs.coding.sandbox import SandboxUnavailable
from oarp.envs.coding.task import load_task
from oarp.envs.coding.tools import READ_LIMIT_BYTES, SYSTEM_PROMPT, coding_tools, task_prompt
from oarp.envs.coding.workspace import Workspace

TASK = load_task(TASKS_DIR / "coding" / "dev" / "off_by_one_001")
SOLUTION = (TASK.solution_dir / "src" / "stats.py").read_text()


@pytest.fixture
def ws():
    with Workspace.create(TASK) as workspace:
        yield workspace


def call(name: str, call_id: str = "c1", **arguments: object) -> ModelResponse:
    return ModelResponse(tool_calls=(ToolCall(call_id, name, dict(arguments)),))


def execute(ws: Workspace, name: str, **arguments: object):
    return ToolRegistry(coding_tools(ws)).execute(ToolCall("c1", name, dict(arguments)))


# --- without Docker --------------------------------------------------------------------


def test_tool_set_and_run_tests_switch(ws: Workspace) -> None:
    assert [t.spec.name for t in coding_tools(ws)] == ["read_file", "write_file", "run_tests"]
    assert [t.spec.name for t in coding_tools(ws, run_tests_enabled=False)] == ["read_file", "write_file"]


def test_read_and_write_through_registry(ws: Workspace) -> None:
    assert "def moving_average" in execute(ws, "read_file", path="src/stats.py").content
    result = execute(ws, "write_file", path="src/stats.py", content=SOLUTION)
    assert not result.is_error and result.content == f"Wrote {len(SOLUTION)} bytes to src/stats.py."
    assert ws.read_text("src/stats.py") == SOLUTION


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        ("write_file", {"path": "tests/test_stats.py", "content": "x"}, "not allowed to write"),
        ("write_file", {"path": "conftest.py", "content": "x"}, "not allowed to write"),
        ("read_file", {"path": "../../.env"}, "outside the workspace"),
        ("write_file", {"path": "src/stats.py"}, "missing required argument(s): content"),
    ],
)
def test_violations_come_back_as_error_results(ws: Workspace, name: str, arguments: dict, expected: str) -> None:
    result = execute(ws, name, **arguments)
    assert result.is_error and expected in result.content


def test_large_file_read_is_truncated(ws: Workspace) -> None:
    (ws.root / "src" / "big.py").write_text("#" * (READ_LIMIT_BYTES + 500))
    content = execute(ws, "read_file", path="src/big.py").content
    assert content.endswith(f"[truncated: showing {READ_LIMIT_BYTES} of {READ_LIMIT_BYTES + 500} bytes]")


def test_prompts(ws: Workspace) -> None:
    prompt = task_prompt(ws)
    assert prompt.startswith(TASK.prompt)
    assert "- src/stats.py (editable)" in prompt
    assert "- tests/test_stats.py (editable)" not in prompt
    assert prompt.endswith("- tests/test_stats.py")
    assert "test" not in SYSTEM_PROMPT.lower()  # neutral about using test feedback
    assert "finish" in SYSTEM_PROMPT


def test_sandbox_outage_aborts_the_run_instead_of_reaching_the_model(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(image: str = sandbox.IMAGE) -> str:
        raise SandboxUnavailable("docker daemon unavailable")

    monkeypatch.setattr(sandbox, "image_id", unavailable)
    model = ScriptedModel([call("run_tests"), call("finish", "c2", answer="never reached")])
    with pytest.raises(SandboxUnavailable):
        run_agent(model, ToolRegistry(coding_tools(ws)), system_prompt=SYSTEM_PROMPT,
                  task_prompt=task_prompt(ws), max_steps=5)
    assert len(model.calls) == 1  # the model never saw an error message about Docker


def test_disabled_run_tests_is_an_unknown_tool(ws: Workspace) -> None:
    registry = ToolRegistry(coding_tools(ws, run_tests_enabled=False))
    result = registry.execute(ToolCall("c1", "run_tests", {}))
    assert result.is_error and "unknown tool 'run_tests'" in result.content


# --- with Docker: the full pipeline, scripted ---------------------------------------------


@pytest.mark.docker
@requires_docker
def test_scripted_agent_fixes_the_bug_end_to_end(ws: Workspace) -> None:
    model = ScriptedModel([
        call("read_file", "c1", path="src/stats.py"),
        call("run_tests", "c2"),
        call("write_file", "c3", path="src/stats.py", content=SOLUTION),
        call("run_tests", "c4"),
        call("finish", "c5", answer="Fixed the off-by-one in moving_average."),
    ])
    result = run_agent(model, ToolRegistry(coding_tools(ws)), system_prompt=SYSTEM_PROMPT,
                       task_prompt=task_prompt(ws), max_steps=10)

    assert result.stop_reason is StopReason.FINISHED
    read, failing, write, passing = (s.tool_results[0] for s in result.steps[:4])
    assert "range(len(values) - window)" in read.content
    # A failing test run is information, not a tool error.
    assert not failing.is_error
    assert failing.content.startswith("Test command exited with code 1.") and "3 failed, 2 passed" in failing.content
    assert not write.is_error
    assert passing.content.startswith("Test command exited with code 0 (all tests passed).")
    assert "5 passed" in passing.content
    assert ws.read_text("src/stats.py") == SOLUTION
    assert (TASK.repo_dir / "src" / "stats.py").read_text() != SOLUTION  # original task untouched
