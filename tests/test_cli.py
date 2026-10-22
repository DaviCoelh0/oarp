"""CLI tests: argument handling without Docker, and free --dry-run / rescore
end to end with Docker. No test here makes a paid call."""

import json
import shutil
import sys
from pathlib import Path

import pytest
from conftest import TASKS_DIR, requires_docker

from oarp.cli import EXIT_FAILURE, EXIT_INVALID, EXIT_SUCCESS, main

TASK = TASKS_DIR / "coding" / "dev" / "off_by_one_001"


def test_command_is_required(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2  # argparse usage error


def test_paid_run_without_key_stops_before_any_call(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    code = main(["run", "--task", str(TASK), "--env-file", str(tmp_path / "missing.env"),
                 "--runs-dir", str(tmp_path / "runs")])
    assert code == EXIT_INVALID
    assert "ANTHROPIC_API_KEY not found" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()  # nothing was started


@pytest.fixture
def dry_run(tmp_path: Path, capsys: pytest.CaptureFixture) -> Path:
    code = main(["run", "--task", str(TASK), "--dry-run", "--runs-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == EXIT_FAILURE  # the scripted dry run changes nothing, so the task fails
    assert "verified success: False (tests_failed" in out and "stop reason:      finished" in out
    (run_dir,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    return run_dir


@pytest.mark.docker
@requires_docker
def test_dry_run_writes_a_complete_run(dry_run: Path) -> None:
    assert {p.name for p in dry_run.iterdir()} == {"trace.jsonl", "summary.json", "final.diff", "final"}
    summary = json.loads((dry_run / "summary.json").read_text())
    assert summary["config"]["model"] == {"model": "scripted-dry-run"}
    assert summary["config"]["harness"] == {"max_steps": 12, "max_tool_calls_per_step": 1,
                                            "run_tests_enabled": True, "loop": "plain"}


@pytest.mark.docker
@requires_docker
def test_rescore_reproduces_the_recorded_result(dry_run: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(["rescore", str(dry_run), "--tasks-dir", str(TASKS_DIR)]) == EXIT_SUCCESS
    assert "file hashes match" in capsys.readouterr().out


@pytest.mark.docker
@requires_docker
def test_rescore_detects_altered_saved_files(dry_run: Path, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    tampered = tmp_path / "tampered"
    shutil.copytree(dry_run, tampered)
    solution = (TASK / "solution" / "src" / "stats.py").read_text()
    (tampered / "final" / "src" / "stats.py").write_text(solution)  # pretend the agent had fixed it
    assert main(["rescore", str(tampered), "--tasks-dir", str(TASKS_DIR)]) == EXIT_FAILURE
    assert "file hashes DIFFER" in capsys.readouterr().out


EXPERIMENTS = TASKS_DIR.parent / "experiments"
COMMITTED_RUNS = sorted(p.parent for p in EXPERIMENTS.glob("*/**/summary.json"))


@pytest.mark.docker
@requires_docker
@pytest.mark.parametrize("run_dir", COMMITTED_RUNS, ids=[p.name for p in COMMITTED_RUNS])
def test_rescore_reproduces_every_committed_run(run_dir: Path, capsys: pytest.CaptureFixture) -> None:
    # Backward compatibility: format-2 support must not change any recorded result.
    assert main(["rescore", str(run_dir), "--tasks-dir", str(TASKS_DIR)]) == EXIT_SUCCESS
    assert "now passed, recorded passed; file hashes match" in capsys.readouterr().out


def test_committed_runs_were_found() -> None:
    assert len(COMMITTED_RUNS) == 16  # 15 P4 runs + 1 CLI run


def test_heldout_task_needs_flag(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    root = tmp_path / "test" / "off_by_one_001"
    shutil.copytree(TASK, root)
    (root / "task.toml").write_text((root / "task.toml").read_text().replace('split = "dev"', 'split = "test"'))
    code = main(["run", "--task", str(root), "--dry-run", "--runs-dir", str(tmp_path / "runs")])
    assert code == EXIT_INVALID
    assert "held-out task; pass --heldout" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_graph_loop_without_extra_stops_before_any_call(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "langgraph.graph", None)  # makes the import fail
    monkeypatch.delitem(sys.modules, "oarp.core.graph_loop", raising=False)
    code = main(["run", "--task", str(TASK), "--loop", "graph", "--env-file", str(tmp_path / "missing.env"),
                 "--runs-dir", str(tmp_path / "runs")])
    assert code == EXIT_INVALID
    assert "pip install -e '.[langgraph]'" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


@pytest.mark.docker
@requires_docker
def test_dry_run_with_graph_loop(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    pytest.importorskip("langgraph")
    code = main(["run", "--task", str(TASK), "--dry-run", "--loop", "graph", "--runs-dir", str(tmp_path)])
    assert code == EXIT_FAILURE  # same outcome as the plain dry run
    assert "stop reason:      finished after 2 model calls" in capsys.readouterr().out
    (run_dir,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["config"]["harness"]["loop"] == "graph"
