"""Independent verifier: decides task success from the agent's final editable
files alone, never from anything the agent reported or ran.

    original repo/  + the final editable files  ->  fresh workspace
    fresh pristine-tests snapshot mounted over /work/tests
    task.test_command in the sandbox            ->  VerificationResult

It is the fixed measuring instrument: identical for every harness
configuration (with or without run_tests), and it runs regardless of how the
agent's run ended (a MAX_STEPS run may still have solved the task).
"""

import difflib
import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum

from oarp.envs.coding.sandbox import run_in_sandbox
from oarp.envs.coding.task import CodingTask
from oarp.envs.coding.workspace import Workspace


class VerificationStatus(StrEnum):
    PASSED = "passed"  # exit 0: success
    TESTS_FAILED = "tests_failed"  # exit 1
    COLLECTION_ERROR = "collection_error"  # exit 2, e.g. the agent's code no longer imports
    TIMEOUT = "timeout"
    KILLED = "killed"  # exit 137, usually out of memory
    HARNESS_ERROR = "harness_error"  # exit 3/4/5/...: our setup is wrong, run is invalid


@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    exit_code: int | None
    counts: dict[str, int] | None  # test outcomes; only when the tests ran (exit 0 or 1)
    file_hashes: dict[str, str]  # sha256 of each verified editable file
    diff: str  # unified diff, original -> final, of the editable files
    output: str  # capped test output, for failure analysis
    duration_s: float
    image_id: str

    @property
    def passed(self) -> bool:
        return self.status is VerificationStatus.PASSED

    @property
    def valid(self) -> bool:
        """False when the measurement itself failed, not the agent."""
        return self.status is not VerificationStatus.HARNESS_ERROR


def final_files(ws: Workspace) -> dict[str, str]:
    """The only thing the verifier takes from a run: the editable files."""
    return {rel: ws.read_text(rel) for rel in ws.task.editable}


def verify(task: CodingTask, files: dict[str, str]) -> VerificationResult:
    """Score `files` (editable path -> content) against the task's original tests.

    Takes plain file contents, so a stored run can be re-scored later from its
    saved files. Raises InfrastructureError (SandboxUnavailable) if Docker is down.
    """
    not_editable = sorted(set(files) - set(task.editable))
    if not_editable:
        raise ValueError(f"verifier only accepts editable files, got: {not_editable}")

    with Workspace.create(task) as fresh:
        originals = {rel: fresh.read_text(rel) for rel in task.editable}
        for rel, content in files.items():
            fresh.write_text(rel, content)
        result = run_in_sandbox(list(task.test_command), fresh.sandbox_mounts(), timeout_s=task.timeout_s)

    final = {rel: files.get(rel, originals[rel]) for rel in task.editable}
    output = result.stdout + (f"\n--- stderr ---\n{result.stderr}" if result.stderr.strip() else "")
    status = status_for(result.exit_code, result.timed_out)
    # Counts describe test outcomes only if pytest actually ran the tests. After a
    # collection error, harness error, timeout or kill, any "N error"/"N warning"
    # in the output describes something else, so it is not reported as counts.
    tests_ran = status in (VerificationStatus.PASSED, VerificationStatus.TESTS_FAILED)
    return VerificationResult(
        status=status,
        exit_code=result.exit_code,
        counts=parse_counts(result.stdout) if tests_ran else None,
        file_hashes={rel: hashlib.sha256(text.encode("utf-8")).hexdigest() for rel, text in final.items()},
        diff=_diff(originals, final),
        output=output,
        duration_s=result.duration_s,
        image_id=result.image_id,
    )


def status_for(exit_code: int | None, timed_out: bool) -> VerificationStatus:
    if timed_out:
        return VerificationStatus.TIMEOUT
    return {
        0: VerificationStatus.PASSED,
        1: VerificationStatus.TESTS_FAILED,
        2: VerificationStatus.COLLECTION_ERROR,
        137: VerificationStatus.KILLED,
    }.get(exit_code, VerificationStatus.HARNESS_ERROR)


_COUNT = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|warnings?|deselected)\b")


def parse_counts(stdout: str) -> dict[str, int] | None:
    """Counts from pytest's summary, which is always the LAST non-empty line,
    e.g. '3 failed, 2 passed in 0.01s'. Earlier lines are never searched: they
    can contain text that merely looks like a summary."""
    lines = stdout.strip().splitlines()
    if not lines:
        return None
    last = lines[-1]
    found = _COUNT.findall(last)
    if not found or " in " not in last:
        return None
    return {kind.rstrip("s") if kind in ("errors", "warnings") else kind: int(n) for n, kind in found}


def _diff(originals: dict[str, str], final: dict[str, str]) -> str:
    return "".join(
        "".join(difflib.unified_diff(
            originals[rel].splitlines(keepends=True), final[rel].splitlines(keepends=True),
            fromfile=f"a/{rel}", tofile=f"b/{rel}",
        ))
        for rel in sorted(final)
    )
