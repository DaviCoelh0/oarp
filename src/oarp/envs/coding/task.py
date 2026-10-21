"""Coding-task definitions: one folder per task, loaded from `task.toml`.

    <task_dir>/
      task.toml     id, split, prompt, editable files, test command, timeout
      repo/         the agent's starting point (src/ and tests/)
      solution/     reference fix for the editable files; never shown to the agent
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

TASK_FILE = "task.toml"
TESTS_DIR = "tests"
SPLITS = ("dev", "test")


class TaskError(ValueError):
    """The task definition itself is invalid (our bug, not the model's)."""


@dataclass(frozen=True)
class CodingTask:
    id: str
    split: str
    prompt: str
    root: Path
    editable: tuple[str, ...]  # repo-relative POSIX paths the agent may overwrite
    test_command: tuple[str, ...]  # run inside the sandbox with /work = repo
    timeout_s: float

    @property
    def repo_dir(self) -> Path:
        return self.root / "repo"

    @property
    def tests_dir(self) -> Path:
        return self.repo_dir / TESTS_DIR

    @property
    def solution_dir(self) -> Path:
        return self.root / "solution"


def load_task(task_dir: Path) -> CodingTask:
    root = task_dir.resolve()
    try:
        data = tomllib.loads((root / TASK_FILE).read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise TaskError(f"{root}: cannot read {TASK_FILE}: {exc}") from exc

    def field(name: str, kind: type | tuple[type, ...]):
        if not isinstance(data.get(name), kind):
            raise TaskError(f"{root}: '{name}' is missing or has the wrong type")
        return data[name]

    task = CodingTask(
        id=field("id", str),
        split=field("split", str),
        prompt=field("prompt", str),
        root=root,
        editable=tuple(field("editable", list)),
        test_command=tuple(field("test_command", list)),
        timeout_s=float(field("timeout_s", (int, float))),
    )
    _validate(task)
    return task


def discover_tasks(base_dir: Path) -> list[CodingTask]:
    """All tasks below `base_dir`, sorted by id."""
    return sorted((load_task(p.parent) for p in base_dir.rglob(TASK_FILE)), key=lambda t: t.id)


def _validate(task: CodingTask) -> None:
    def fail(message: str) -> None:
        raise TaskError(f"task '{task.id}': {message}")

    if task.split not in SPLITS:
        fail(f"split must be one of {SPLITS}")
    if task.id != task.root.name:
        fail(f"id must match its folder name '{task.root.name}'")
    if not task.tests_dir.is_dir():
        fail(f"missing {TESTS_DIR}/ in repo/")
    if not task.editable or not all(isinstance(p, str) for p in task.editable):
        fail("editable must be a non-empty list of paths")
    if not task.test_command or not all(isinstance(c, str) for c in task.test_command):
        fail("test_command must be a non-empty list of strings")
    if task.timeout_s <= 0:
        fail("timeout_s must be positive")
    # Symlinks in a task could point outside the workspace once copied.
    if any(p.is_symlink() for p in task.root.rglob("*")):
        fail("task folders must not contain symlinks")
    for rel in task.editable:
        pure = PurePosixPath(rel)
        if pure.is_absolute() or ".." in pure.parts:
            fail(f"editable path must be relative without '..': {rel}")
        if pure.parts[0] == TESTS_DIR:
            fail(f"tests can never be editable: {rel}")
        if not (task.repo_dir / pure).is_file():
            fail(f"editable file does not exist in repo/: {rel}")
    if task.solution_dir.exists():
        solution_files = {
            p.relative_to(task.solution_dir).as_posix() for p in task.solution_dir.rglob("*") if p.is_file()
        }
        if not solution_files <= set(task.editable):
            fail(f"solution/ may only contain editable files: {sorted(solution_files - set(task.editable))}")
