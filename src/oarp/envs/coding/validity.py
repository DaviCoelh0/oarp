"""Task validity checks, run before a task is used in any paid run.

- static_problems: no Docker. The starting repo must not leak the reference
  or a decoy, and the reference must actually change something.
- behaviour_problems: needs Docker, judged by the verifier itself. The
  original fails the public tests (exactly 'tests_failed'), the reference
  passes public and hidden tests, every decoy fails, and at least one decoy
  fails the public tests (so the mistake is visible to the agent, by reading
  or by running the tests).

Each function returns a list of problems; empty means valid.
"""

from oarp.envs.coding.task import CodingTask
from oarp.envs.coding.verify import VerificationStatus as S, verify

# Words that would point the agent at the evaluation material.
LEAK_WORDS = ("solution", "decoy", "hidden")


def candidate(task: CodingTask, folder: str) -> dict[str, str]:
    """Editable-file contents of solution/ or decoys/<name>/ (only the files it has)."""
    root = task.root / folder
    return {rel: (root / rel).read_text() for rel in task.editable if (root / rel).is_file()}


def static_problems(task: CodingTask) -> list[str]:
    problems = []
    repo_files = {p.relative_to(task.repo_dir).as_posix(): p.read_bytes()
                  for p in task.repo_dir.rglob("*") if p.is_file()}

    for word in LEAK_WORDS:
        if word in task.prompt.lower():
            problems.append(f"prompt mentions '{word}'")
        for rel, data in repo_files.items():
            if word in rel.lower() or word in data.decode("utf-8", errors="ignore").lower():
                problems.append(f"repo/{rel} mentions '{word}'")

    # No candidate file may appear verbatim in the starting repo: that would
    # either leak it or mean the candidate changes nothing.
    candidates = {"solution": candidate(task, "solution")} if task.solution_dir.is_dir() else {}
    candidates.update({f"decoy {n}": candidate(task, f"decoys/{n}") for n in task.decoy_names()})
    for label, files in candidates.items():
        if not files:
            problems.append(f"{label} has no files")
        for rel, text in files.items():
            for repo_rel, data in repo_files.items():
                if data == text.encode("utf-8"):
                    problems.append(f"{label} file {rel} is identical to repo/{repo_rel}")
    return problems


def behaviour_problems(task: CodingTask) -> list[str]:
    problems = []
    original = verify(task, {})
    if original.public.status is not S.TESTS_FAILED:
        problems.append(f"original: public tests gave {original.public.status}, expected tests_failed")

    reference = verify(task, candidate(task, "solution"))
    if not reference.passed:
        problems.append(f"reference: {reference.status} (public {reference.public.status}, "
                        f"hidden {reference.hidden.status if reference.hidden else '-'})")

    visible_decoy = False
    for name in task.decoy_names():
        result = verify(task, candidate(task, f"decoys/{name}"))
        if result.status is not S.TESTS_FAILED:
            problems.append(f"decoy {name}: {result.status}, expected tests_failed")
        visible_decoy |= result.public.status is S.TESTS_FAILED
    if task.decoy_names() and not visible_decoy:
        problems.append("no decoy fails the public tests")
    return problems
