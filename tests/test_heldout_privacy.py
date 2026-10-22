"""What the agent can see, what gets written where, and when held-out tasks
may run (invariants 1, 2, 8, 13, 14, 15 of the M2 task design)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import FIXTURES_DIR, requires_docker

from oarp.core.model import ScriptedModel
from oarp.core.tools import ToolError
from oarp.core.types import ModelResponse, ToolCall
from oarp.envs.coding.runner import HarnessConfig, HeldOutError, run_coding_task
from oarp.envs.coding.task import load_task, task_hash
from oarp.envs.coding.tools import coding_tools
from oarp.envs.coding.workspace import Workspace

F2 = FIXTURES_DIR / "count_words_f2"
TASK = load_task(F2)
SENTINEL = "HIDDEN_SENTINEL_7f3a"
OVERFIT = (F2 / "decoys" / "drop_empty_pieces" / "src" / "words.py").read_text()
REPO_ROOT = Path(__file__).resolve().parents[1]


def call(name: str, call_id: str, **arguments: object) -> ModelResponse:
    return ModelResponse(tool_calls=(ToolCall(call_id, name, dict(arguments)),))


def test_workspace_never_contains_evaluation_material() -> None:
    with Workspace.create(TASK) as ws:
        assert ws.list_files() == ["src/words.py", "tests/test_words.py"]
        for path in ["../hidden_tests/test_words_hidden.py", "../pristine_tests/test_words.py",
                     str(F2 / "hidden_tests" / "test_words_hidden.py"), "../../solution/src/words.py"]:
            with pytest.raises(ToolError):
                ws.read_text(path)
        with pytest.raises(ToolError):
            ws.write_text("../hidden_tests/test_words_hidden.py", "")


@pytest.mark.docker
@requires_docker
def test_run_tests_runs_only_public_tests() -> None:
    with Workspace.create(TASK) as ws:
        run_tests = next(t for t in coding_tools(ws) if t.spec.name == "run_tests")
        output = run_tests.handler()
    assert "2 failed, 1 passed" in output  # the 3 public tests, 2 failing on the original
    assert SENTINEL not in output and "test_words_hidden" not in output


@pytest.fixture
def overfit_run(tmp_path: Path) -> tuple[dict, Path, ScriptedModel]:
    """A scripted agent that reads everything it can, runs the tests and writes
    a fix that passes the public tests but fails a hidden one."""
    model = ScriptedModel([
        call("read_file", "c1", path="src/words.py"),
        call("read_file", "c2", path="tests/test_words.py"),
        call("run_tests", "c3"),
        call("write_file", "c4", path="src/words.py", content=OVERFIT),
        call("run_tests", "c5"),
        call("finish", "c6", answer="All visible tests pass."),
    ])
    summary = run_coding_task(TASK, model, HarnessConfig(max_steps=12), model_settings={"model": "scripted"},
                              runs_dir=tmp_path)
    return summary, tmp_path / summary["run_id"], model


@pytest.mark.docker
@requires_docker
def test_overfit_run_is_scored_as_failure_with_parts(overfit_run) -> None:
    summary, _, _ = overfit_run
    assert summary["valid"] and not summary["verified_success"]
    assert (summary["public_status"], summary["hidden_status"]) == ("passed", "tests_failed")
    assert (summary["test_counts"], summary["hidden_counts"]) == ({"passed": 3}, {"failed": 1, "passed": 1})
    assert summary["task_format"] == 2 and summary["task_hash"] == task_hash(TASK)
    assert summary["heldout_opt_in"] is False


@pytest.mark.docker
@requires_docker
def test_agent_never_sees_evaluation_material(overfit_run) -> None:
    _, _, model = overfit_run
    seen = repr(model.calls)  # every message and tool spec the model was shown
    hidden_text = (F2 / "hidden_tests" / "test_words_hidden.py").read_text()
    solution_text = (F2 / "solution" / "src" / "words.py").read_text()
    assert SENTINEL not in seen and "test_words_hidden" not in seen
    assert hidden_text not in seen and solution_text not in seen
    for word in ("hidden_tests", "solution", "decoys", "strip_then_split"):
        assert word not in seen


@pytest.mark.docker
@requires_docker
def test_hidden_output_is_only_in_private(overfit_run) -> None:
    _, run_dir, _ = overfit_run
    assert SENTINEL in (run_dir / "private" / "verifier_hidden.txt").read_text()
    for path in run_dir.rglob("*"):
        if path.is_file() and "private" not in path.relative_to(run_dir).parts:
            assert SENTINEL not in path.read_text(), path.name
    verification = [json.loads(line) for line in (run_dir / "trace.jsonl").read_text().splitlines()][-1]
    assert verification["event"] == "verification"
    assert verification["hidden_status"] == "tests_failed" and verification["hidden_counts"] == {"failed": 1, "passed": 1}


def heldout_copy(tmp_path: Path) -> Path:
    root = tmp_path / "test" / "count_words_f2"
    shutil.copytree(F2, root)
    toml = root / "task.toml"
    toml.write_text(toml.read_text().replace('split = "dev"', 'split = "test"'))
    return root


def test_heldout_task_refused_without_opt_in(tmp_path: Path) -> None:
    task = load_task(heldout_copy(tmp_path))
    model = ScriptedModel([])  # would raise if it were ever called
    with pytest.raises(HeldOutError, match="allow_heldout"):
        run_coding_task(task, model, HarnessConfig(), model_settings={"model": "scripted"},
                        runs_dir=tmp_path / "runs")
    assert model.calls == [] and not (tmp_path / "runs").exists()


@pytest.mark.docker
@requires_docker
def test_heldout_task_runs_with_opt_in_and_records_it(tmp_path: Path) -> None:
    task = load_task(heldout_copy(tmp_path))
    summary = run_coding_task(task, ScriptedModel([call("finish", "c1", answer="x")]), HarnessConfig(),
                              model_settings={"model": "scripted"}, runs_dir=tmp_path / "runs",
                              allow_heldout=True)
    assert summary["split"] == "test" and summary["heldout_opt_in"] is True and summary["valid"]


@pytest.mark.parametrize("path", ["tasks/coding/test/some_task/task.toml",
                                  "runs/20261022T000000-abcdef/private/verifier_hidden.txt",
                                  "experiments/m2/results/r1/private/verifier_hidden.txt"])
def test_gitignore_keeps_heldout_and_private_out(path: str) -> None:
    result = subprocess.run(["git", "-C", str(REPO_ROOT), "check-ignore", "-q", path])
    assert result.returncode == 0, f"{path} is not ignored"


def test_gitignore_does_not_hide_dev_tasks() -> None:
    result = subprocess.run(["git", "-C", str(REPO_ROOT), "check-ignore", "-q",
                             "tasks/coding/dev/off_by_one_001/task.toml"])
    assert result.returncode == 1
