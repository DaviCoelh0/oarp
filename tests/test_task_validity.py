"""Every task must be a real, solvable problem, judged by the verifier itself:
the original code fails its public tests (exactly 'tests_failed', not a
collection or harness error), the reference passes public and hidden tests,
every decoy fails, and at least one decoy fails the public tests. The starting
repo must not leak the reference or a decoy.

Held-out tasks are included when present locally: these checks run no model.
"""

import pytest
from conftest import FIXTURES_DIR, TASKS_DIR, requires_docker

from oarp.envs.coding.task import discover_tasks, load_task
from oarp.envs.coding.validity import behaviour_problems, static_problems

TASKS = [*discover_tasks(TASKS_DIR, include_heldout=True), load_task(FIXTURES_DIR / "count_words_f2")]


@pytest.mark.parametrize("task", TASKS, ids=[t.id for t in TASKS])
def test_no_leaks_in_starting_repo(task) -> None:
    assert task.solution_dir.is_dir(), "every task needs a reference solution"
    assert static_problems(task) == []


@pytest.mark.docker
@requires_docker
@pytest.mark.parametrize("task", TASKS, ids=[t.id for t in TASKS])
def test_original_fails_reference_passes_decoys_fail(task) -> None:
    assert behaviour_problems(task) == []
