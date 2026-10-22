"""End-to-end runs with a scripted model: workspace -> tools -> loop -> verifier
-> trace files. Deterministic and free; needs Docker for the tests and verifier."""

import json
import re
import sys
from pathlib import Path

import pytest
from conftest import TASKS_DIR, requires_docker

from oarp.core.model import ScriptedModel
from oarp.core.types import ModelResponse, ToolCall, Usage
from oarp.envs.coding import sandbox
from oarp.envs.coding.runner import HarnessConfig, run_coding_task
from oarp.envs.coding.task import load_task
from oarp.envs.coding.verify import verify

TASK = load_task(TASKS_DIR / "coding" / "dev" / "off_by_one_001")
SOLUTION = (TASK.solution_dir / "src" / "stats.py").read_text()
HAIKU = "claude-haiku-5-5"


def call(name: str, call_id: str, **arguments: object) -> ModelResponse:
    # Pretend to be Haiku so cost estimation is exercised.
    return ModelResponse(tool_calls=(ToolCall(call_id, name, dict(arguments)),),
                         usage=Usage(1000, 100), model=HAIKU, stop_reason="tool_use")


FIX = [
    call("read_file", "c1", path="src/stats.py"),
    call("run_tests", "c2"),
    call("write_file", "c3", path="src/stats.py", content=SOLUTION),
    call("run_tests", "c4"),
]


def run(script: list[ModelResponse], tmp_path: Path, **harness) -> tuple[dict, Path]:
    summary = run_coding_task(TASK, ScriptedModel(script), HarnessConfig(**harness),
                              model_settings={"model": "scripted"}, runs_dir=tmp_path)
    return summary, tmp_path / summary["run_id"]


def events(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "trace.jsonl").read_text().splitlines()]


@pytest.mark.docker
@requires_docker
def test_successful_run_is_fully_recorded(tmp_path: Path) -> None:
    summary, run_dir = run([*FIX, call("finish", "c5", answer="Fixed the range bound.")], tmp_path)

    assert summary["valid"] and summary["verified_success"]
    assert (summary["stop_reason"], summary["verification_status"]) == ("finished", "passed")
    assert summary["test_counts"] == {"passed": 5}
    assert (summary["model_calls"], summary["tool_calls_attempted"], summary["run_tests_calls"]) == (5, 4, 2)
    assert summary["used_test_feedback"] and summary["errors_by_kind"] == {}
    assert (summary["input_tokens"], summary["output_tokens"]) == (5000, 500)
    assert summary["estimated_cost_usd"] == pytest.approx(5 * (1000 * 0.10e-6 + 100 * 0.50e-6))
    assert summary["sandbox_image"].startswith("sha256:")

    kinds = [e["event"] for e in events(run_dir)]
    assert kinds[0] == "run_start" and kinds[-2:] == ["run_end", "verification"]
    assert kinds.count("model_call") == 5 and kinds.count("tool_result") == 4
    start = events(run_dir)[0]
    assert set(start["prompt_hashes"]) == {"system", "task", "tools"}
    assert len(start["provenance"]["git"]["commit"]) == 40
    assert json.loads((run_dir / "summary.json").read_text()) == json.loads(json.dumps(summary, default=str))

    # Saved final files can be re-scored independently, with the same result.
    saved = {"src/stats.py": (run_dir / "final" / "src" / "stats.py").read_text()}
    assert verify(TASK, saved).file_hashes == events(run_dir)[-1]["file_hashes"]
    assert "+    return" in (run_dir / "final.diff").read_text()


@pytest.mark.docker
@requires_docker
def test_solving_without_finish_is_max_steps_but_verified_success(tmp_path: Path) -> None:
    summary, _ = run(FIX, tmp_path, max_steps=4)
    assert summary["stop_reason"] == "max_steps"
    assert summary["verified_success"] is True  # termination and success are separate


@pytest.mark.docker
@requires_docker
def test_claiming_success_without_fixing_is_verified_failure(tmp_path: Path) -> None:
    summary, _ = run([call("finish", "c1", answer="All tests pass now!")], tmp_path)
    assert summary["stop_reason"] == "finished"
    assert summary["verified_success"] is False and summary["verification_status"] == "tests_failed"
    assert summary["test_counts"] == {"failed": 3, "passed": 2}
    assert not summary["used_test_feedback"]


@pytest.mark.docker
@requires_docker
def test_error_kinds_are_counted(tmp_path: Path) -> None:
    script = [
        call("write_file", "c1", path="tests/test_stats.py", content="def test_x(): pass\n"),
        call("delete_everything", "c2"),
        call("finish", "c3", answer="gave up"),
    ]
    summary, _ = run(script, tmp_path)
    assert summary["errors_by_kind"] == {"tool_refused": 1, "unknown_tool": 1}
    assert summary["verified_success"] is False


@pytest.mark.docker
@requires_docker
def test_disabled_run_tests_is_recorded_in_config(tmp_path: Path) -> None:
    without, without_dir = run([call("finish", "c1", answer="x")], tmp_path, run_tests_enabled=False)
    with_tests, with_dir = run([call("finish", "c1", answer="x")], tmp_path)
    assert without["config"]["harness"]["run_tests_enabled"] is False
    assert without["config_hash"] != with_tests["config_hash"]
    hashes_without = events(without_dir)[0]["prompt_hashes"]
    hashes_with = events(with_dir)[0]["prompt_hashes"]
    assert hashes_without["tools"] != hashes_with["tools"]  # run_tests absent from the tool list
    assert hashes_without["system"] == hashes_with["system"]  # the prompt is NOT changed by the switch


def test_infrastructure_failure_marks_run_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(image: str = sandbox.IMAGE) -> str:
        raise sandbox.SandboxUnavailable("docker daemon unavailable")

    monkeypatch.setattr(sandbox, "image_id", unavailable)
    summary, run_dir = run([call("run_tests", "c1"), call("finish", "c2", answer="x")], tmp_path)
    assert summary["valid"] is False and "SandboxUnavailable" in summary["invalid_reason"]
    assert "verified_success" not in summary  # never scored
    kinds = [e["event"] for e in events(run_dir)]
    assert kinds == ["run_start", "model_call", "run_invalid"]  # partial trace survives


def test_no_secrets_in_run_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-FAKE-SECRET-FOR-TEST")
    monkeypatch.setattr(sandbox, "image_id", lambda image=sandbox.IMAGE: (_ for _ in ()).throw(
        sandbox.SandboxUnavailable("no docker in this test")))
    _, run_dir = run([call("read_file", "c1", path="src/stats.py"), call("run_tests", "c2")], tmp_path)
    for path in run_dir.rglob("*"):
        if path.is_file():
            text = path.read_text()
            assert "sk-ant" not in text and "FAKE-SECRET" not in text


# Fields that legitimately differ between two runs of the same script.
VOLATILE = {"ts", "t_s", "latency_s", "duration_s", "run_id", "config", "config_hash", "wall_time_s"}


PYTEST_DURATION = re.compile(r" in \d+\.\d+s\b")  # "5 passed in 0.01s" varies run to run


def comparable(record: dict) -> dict:
    return {k: PYTEST_DURATION.sub(" in <t>s", v) if k in ("content", "output") else v
            for k, v in record.items() if k not in VOLATILE}


@pytest.mark.docker
@requires_docker
def test_graph_loop_run_matches_plain_run(tmp_path: Path) -> None:
    pytest.importorskip("langgraph")
    script = [*FIX, call("finish", "c5", answer="Fixed the range bound.")]
    plain, plain_dir = run(script, tmp_path)
    graph, graph_dir = run(script, tmp_path, loop="graph")

    assert graph["valid"] and graph["verified_success"] and graph["stop_reason"] == "finished"
    assert (plain["config"]["harness"]["loop"], graph["config"]["harness"]["loop"]) == ("plain", "graph")
    assert plain["config_hash"] != graph["config_hash"]  # the loop implementation is part of the config
    assert comparable(graph) == comparable(plain)
    assert [comparable(e) for e in events(graph_dir)] == [comparable(e) for e in events(plain_dir)]
    assert events(graph_dir)[0]["provenance"]["langgraph"] is not None


def test_graph_loop_without_extra_fails_before_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "langgraph.graph", None)  # makes the import fail
    monkeypatch.delitem(sys.modules, "oarp.core.graph_loop", raising=False)
    with pytest.raises(ImportError, match=r"\[langgraph\]"):
        run([call("finish", "c1", answer="x")], tmp_path, loop="graph")
    assert list(tmp_path.iterdir()) == []


def test_unknown_loop_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown loop"):
        run([call("finish", "c1", answer="x")], tmp_path, loop="dag")
