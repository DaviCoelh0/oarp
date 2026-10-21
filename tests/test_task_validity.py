"""Every task must be a real, solvable problem, judged by the verifier itself:
the original code fails its tests (exactly 'tests_failed', not a collection or
harness error), and the reference solution passes.
"""

import pytest
from conftest import TASKS_DIR, requires_docker

from oarp.envs.coding.task import discover_tasks
from oarp.envs.coding.verify import VerificationStatus, verify

TASKS = discover_tasks(TASKS_DIR)


@pytest.mark.docker
@requires_docker
@pytest.mark.parametrize("task", TASKS, ids=[t.id for t in TASKS])
def test_original_fails_and_solution_passes(task) -> None:
    assert task.solution_dir.is_dir(), "every task needs a reference solution"

    before = verify(task, {})
    assert before.status is VerificationStatus.TESTS_FAILED, before.output

    solution = {rel: (task.solution_dir / rel).read_text() for rel in task.editable if (task.solution_dir / rel).exists()}
    after = verify(task, solution)
    assert after.status is VerificationStatus.PASSED, after.output
