"""The M2 pilot analyzer must count test feedback correctly before it is used
on paid runs. A scripted agent first writes a wrong fix, sees the tests fail,
then writes the reference: every metric has a known expected value."""

import importlib.util
from pathlib import Path

import pytest
from conftest import TASKS_DIR, requires_docker

from oarp.core.model import ScriptedModel
from oarp.core.types import ModelResponse, ToolCall
from oarp.envs.coding.runner import HarnessConfig, run_coding_task
from oarp.envs.coding.task import load_task

SPEC = importlib.util.spec_from_file_location(
    "m2_pilot_analyze", TASKS_DIR.parent / "experiments" / "m2_pilot" / "analyze.py")
analyze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyze)

TASK = load_task(TASKS_DIR / "coding" / "dev" / "config_merge_007")
LOADER = "src/appconfig/loader.py"
WRONG = (TASK.decoys_dir / "deepcopy_result" / LOADER).read_text()
RIGHT = (TASK.solution_dir / LOADER).read_text()


def call(name: str, call_id: str, **arguments: object) -> ModelResponse:
    return ModelResponse(tool_calls=(ToolCall(call_id, name, dict(arguments)),))


@pytest.mark.parametrize(("content", "code"), [
    ("Test command exited with code 0 (all tests passed).\n--- stdout ---", 0),
    ("Test command exited with code 1.\n--- stdout ---\n1 failed", 1),
    ("Tests timed out after 30s; the process was killed.", None),
])
def test_exit_code_parsing(content: str, code) -> None:
    assert analyze.test_exit_code(content) == code


@pytest.mark.docker
@requires_docker
def test_recovery_run_is_counted_exactly(tmp_path: Path) -> None:
    script = [
        call("read_file", "c1", path=LOADER),
        call("write_file", "c2", path=LOADER, content=WRONG),
        call("run_tests", "c3"),
        call("write_file", "c4", path=LOADER, content=RIGHT),
        call("run_tests", "c5"),
        call("finish", "c6", answer="fixed"),
    ]
    summary = run_coding_task(TASK, ScriptedModel(script), HarnessConfig(max_steps=12),
                              model_settings={"model": "scripted"}, runs_dir=tmp_path)
    (run,) = analyze.load_runs(tmp_path)
    row = analyze.analyse_run(run, {TASK.id: TASK})

    assert summary["verified_success"]
    assert (row["tests_run"], row["failing_seen"], row["edits_after_failure"], row["writes"]) == (2, 1, 1, 2)
    assert row["candidate"] == {"status": "tests_failed", "public": "tests_failed", "hidden": "tests_failed"}
    report = analyze.report([row], [])
    assert "| config_merge_007 | 1/1 | 0/1 | 0/1 | 1/1 | 0/1 | 6 |" in report


@pytest.mark.docker
@requires_docker
def test_run_without_tests_has_no_candidate(tmp_path: Path) -> None:
    script = [call("write_file", "c1", path=LOADER, content=RIGHT), call("finish", "c2", answer="fixed")]
    run_coding_task(TASK, ScriptedModel(script), HarnessConfig(max_steps=12),
                    model_settings={"model": "scripted"}, runs_dir=tmp_path)
    (run,) = analyze.load_runs(tmp_path)
    row = analyze.analyse_run(run, {TASK.id: TASK})
    assert row["candidate"] is None and (row["tests_run"], row["writes"]) == (0, 1)
