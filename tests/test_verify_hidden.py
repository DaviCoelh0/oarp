"""Verifier with hidden tests (format 2): two runs on the same final files,
reported separately; success needs both; a harness error in either part makes
the measurement invalid."""

import pytest
from conftest import FIXTURES_DIR, TASKS_DIR, requires_docker

from oarp.envs.coding import verify as verify_module
from oarp.envs.coding.sandbox import SandboxResult
from oarp.envs.coding.task import load_task
from oarp.envs.coding.verify import VerificationStatus as S, verify

TASK = load_task(FIXTURES_DIR / "count_words_f2")
SOLUTION = (TASK.solution_dir / "src" / "words.py").read_text()
OVERFIT = (TASK.decoys_dir / "drop_empty_pieces" / "src" / "words.py").read_text()


def stub_sandbox(monkeypatch: pytest.MonkeyPatch, exit_codes: list[int]) -> list[list]:
    """Replace the sandbox: the n-th call exits with exit_codes[n]; records the mounts."""
    mounts: list[list] = []

    def fake(command, run_mounts, **kwargs):
        mounts.append(run_mounts)
        code = exit_codes[len(mounts) - 1]
        return SandboxResult(code, f"run {len(mounts)}\n1 passed in 0.01s", "", False, False, 0.1, "sha256:x")

    monkeypatch.setattr(verify_module, "run_in_sandbox", fake)
    return mounts


@pytest.mark.parametrize(
    ("public", "hidden", "overall"),
    [
        (0, 0, S.PASSED),
        (0, 1, S.TESTS_FAILED),  # passes public, fails hidden: overfitting, not success
        (1, 0, S.TESTS_FAILED),
        (1, 4, S.HARNESS_ERROR),  # a broken hidden run invalidates the whole measurement
        (4, 0, S.HARNESS_ERROR),
        (2, 1, S.COLLECTION_ERROR),
    ],
)
def test_overall_status_combines_both_parts(monkeypatch, public: int, hidden: int, overall: S) -> None:
    stub_sandbox(monkeypatch, [public, hidden])
    result = verify(TASK, {"src/words.py": SOLUTION})
    assert result.status is overall
    assert result.passed is (overall is S.PASSED)
    assert result.valid is (overall is not S.HARNESS_ERROR)


def test_both_parts_always_run_on_the_same_files(monkeypatch) -> None:
    mounts = stub_sandbox(monkeypatch, [1, 0])
    result = verify(TASK, {"src/words.py": SOLUTION})
    assert len(mounts) == 2  # hidden tests still run after a public failure
    (public_work, public_tests), (hidden_work, hidden_tests) = mounts
    assert public_work.target == hidden_work.target == "/work"
    assert public_tests.target == hidden_tests.target == "/work/tests"
    assert public_work.source != hidden_work.source  # a fresh workspace per run
    assert (result.public.output, result.hidden.output) == ("run 1\n1 passed in 0.01s", "run 2\n1 passed in 0.01s")
    assert result.output == result.public.output  # the legacy field is the public part only


def test_format1_runs_once(monkeypatch) -> None:
    task = load_task(TASKS_DIR / "coding" / "dev" / "off_by_one_001")
    mounts = stub_sandbox(monkeypatch, [0])
    result = verify(task, {})
    assert len(mounts) == 1 and result.hidden is None and result.status is S.PASSED


@pytest.mark.docker
@requires_docker
def test_overfit_fix_passes_public_but_fails_hidden() -> None:
    result = verify(TASK, {"src/words.py": OVERFIT})
    assert result.public.status is S.PASSED and result.public.counts == {"passed": 3}
    assert result.hidden.status is S.TESTS_FAILED and result.hidden.counts == {"failed": 1, "passed": 1}
    assert result.status is S.TESTS_FAILED and not result.passed and result.valid
    assert "HIDDEN_SENTINEL_7f3a" in result.hidden.output
    assert "HIDDEN_SENTINEL_7f3a" not in result.output


@pytest.mark.docker
@requires_docker
def test_reference_passes_both_parts() -> None:
    result = verify(TASK, {"src/words.py": SOLUTION})
    assert result.passed
    assert (result.public.counts, result.hidden.counts) == ({"passed": 3}, {"passed": 2})
