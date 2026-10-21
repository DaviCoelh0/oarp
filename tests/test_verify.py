from dataclasses import replace

import pytest
from conftest import TASKS_DIR, requires_docker

from oarp.envs.coding.task import load_task
from oarp.envs.coding.verify import (
    VerificationStatus as S,
    final_files,
    parse_counts,
    status_for,
    verify,
)
from oarp.envs.coding.workspace import Workspace

TASK = load_task(TASKS_DIR / "coding" / "dev" / "off_by_one_001")
ORIGINAL = (TASK.repo_dir / "src" / "stats.py").read_text()
SOLUTION = (TASK.solution_dir / "src" / "stats.py").read_text()

# --- pure parts (no Docker) -------------------------------------------------------------


@pytest.mark.parametrize(
    ("exit_code", "timed_out", "status"),
    [
        (0, False, S.PASSED), (1, False, S.TESTS_FAILED), (2, False, S.COLLECTION_ERROR),
        (137, False, S.KILLED), (None, True, S.TIMEOUT),
        (3, False, S.HARNESS_ERROR), (4, False, S.HARNESS_ERROR), (5, False, S.HARNESS_ERROR),
    ],
)
def test_exit_code_mapping(exit_code, timed_out, status) -> None:
    assert status_for(exit_code, timed_out) is status


@pytest.mark.parametrize(
    ("stdout", "counts"),
    [
        ("FFF..   [100%]\n... \n3 failed, 2 passed in 0.01s", {"failed": 3, "passed": 2}),
        (".....  [100%]\n5 passed in 0.00s\n", {"passed": 5}),
        ("E   SyntaxError\n1 error in 0.05s", {"error": 1}),
        ("2 failed, 1 passed, 3 warnings in 0.2s", {"failed": 2, "passed": 1, "warning": 3}),
        ("no summary here", None),
        ("", None),
        # A summary-like line that is NOT last is ignored (the harness-error case).
        ("1 warning in 0.00s\nERROR: file or directory not found: no_such_dir", None),
    ],
)
def test_parse_counts(stdout: str, counts) -> None:
    assert parse_counts(stdout) == counts


def test_non_editable_files_are_refused() -> None:
    with pytest.raises(ValueError, match="only accepts editable files"):
        verify(TASK, {"tests/test_stats.py": "def test_x(): pass\n"})


def test_final_files_takes_only_editable_files() -> None:
    with Workspace.create(TASK) as ws:
        (ws.root / "tests" / "test_stats.py").write_text("def test_fake(): pass\n")  # tampering
        assert final_files(ws) == {"src/stats.py": ORIGINAL}


# --- in the sandbox ----------------------------------------------------------------------


@pytest.mark.docker
@requires_docker
def test_original_code_fails() -> None:
    result = verify(TASK, {"src/stats.py": ORIGINAL})
    assert result.status is S.TESTS_FAILED and not result.passed and result.valid
    assert result.counts == {"failed": 3, "passed": 2}
    assert result.diff == ""
    assert result.image_id.startswith("sha256:")


@pytest.mark.docker
@requires_docker
def test_solution_passes_and_is_recorded() -> None:
    result = verify(TASK, {"src/stats.py": SOLUTION})
    assert result.passed and result.counts == {"passed": 5}
    assert "-    return [sum(values[i : i + window]) / window for i in range(len(values) - window)]" in result.diff
    assert "+    return [sum(values[i : i + window]) / window for i in range(len(values) - window + 1)]" in result.diff
    assert len(result.file_hashes["src/stats.py"]) == 64


@pytest.mark.docker
@requires_docker
def test_no_files_means_original_code() -> None:
    assert verify(TASK, {}).status is S.TESTS_FAILED


@pytest.mark.docker
@requires_docker
def test_broken_code_is_a_collection_error() -> None:
    result = verify(TASK, {"src/stats.py": "def moving_average(:\n"})
    assert result.status is S.COLLECTION_ERROR and result.valid and not result.passed
    assert result.counts is None  # "1 error" counts the import failure, not test outcomes


@pytest.mark.docker
@requires_docker
def test_infinite_loop_times_out() -> None:
    result = verify(replace(TASK, timeout_s=3), {"src/stats.py": "while True:\n    pass\n"})
    assert result.status is S.TIMEOUT and result.valid
    assert result.counts is None


@pytest.mark.docker
@requires_docker
def test_misconfigured_test_command_is_a_harness_error() -> None:
    broken = replace(TASK, test_command=("python", "-m", "pytest", "-q", "no_such_dir"))
    result = verify(broken, {"src/stats.py": SOLUTION})
    assert result.status is S.HARNESS_ERROR and not result.valid
    assert result.counts is None  # previously picked up a stray "1 warning"
