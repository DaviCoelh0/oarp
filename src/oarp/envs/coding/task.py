"""Coding-task definitions: one folder per task, loaded from `task.toml`.

    <task_dir>/
      task.toml       id, split, prompt, editable files, test command, timeout
                      (format 2 adds: format = 2, source = [bug-report URLs])
      repo/           the agent's starting point (src/ and tests/ = public tests)
      solution/       reference fix for the editable files; never shown to the agent
      hidden_tests/   format 2: verifier-only tests; never copied into the workspace
      decoys/<name>/  format 2: deliberately wrong candidate fixes (editable files only)

Three evaluation levels, kept apart on purpose:
- public tests (repo/tests/): the agent can read and run them;
- hidden tests of dev tasks: the agent never sees them, but selection and
  tuning may use their scores ("hidden" is not "held out");
- held-out tasks (split "test"): not loaded or run unless explicitly asked
  for, and never used for selection or tuning.
"""

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

TASK_FILE = "task.toml"
TESTS_DIR = "tests"
HIDDEN_TESTS_DIR = "hidden_tests"
DECOYS_DIR = "decoys"
SPLITS = ("dev", "test")
HELDOUT_SPLIT = "test"
FORMATS = (1, 2)
# Local clutter that must not change a task's hash.
HASH_IGNORED = {"__pycache__", ".DS_Store", ".pytest_cache"}


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
    format: int = 1
    sources: tuple[str, ...] = ()  # bug reports the task's mechanism is adapted from

    @property
    def heldout(self) -> bool:
        return self.split == HELDOUT_SPLIT

    @property
    def repo_dir(self) -> Path:
        return self.root / "repo"

    @property
    def tests_dir(self) -> Path:
        return self.repo_dir / TESTS_DIR

    @property
    def solution_dir(self) -> Path:
        return self.root / "solution"

    @property
    def hidden_tests_dir(self) -> Path | None:
        """Verifier-only tests, or None for format-1 tasks."""
        return self.root / HIDDEN_TESTS_DIR if self.format >= 2 else None

    @property
    def decoys_dir(self) -> Path:
        return self.root / DECOYS_DIR

    def decoy_names(self) -> list[str]:
        return sorted(p.name for p in self.decoys_dir.iterdir() if p.is_dir()) if self.decoys_dir.is_dir() else []


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

    source = data.get("source", [])
    if not isinstance(source, list):
        raise TaskError(f"{root}: 'source' must be a list of URLs")
    task = CodingTask(
        id=field("id", str),
        split=field("split", str),
        prompt=field("prompt", str),
        root=root,
        editable=tuple(field("editable", list)),
        test_command=tuple(field("test_command", list)),
        timeout_s=float(field("timeout_s", (int, float))),
        format=data.get("format", 1),
        sources=tuple(source),
    )
    _validate(task)
    return task


def discover_tasks(base_dir: Path, *, include_heldout: bool = False) -> list[CodingTask]:
    """All tasks below `base_dir`, sorted by id. Held-out tasks are skipped
    unless `include_heldout` is set, so they can't slip into a pilot or a search."""
    tasks = (load_task(p.parent) for p in base_dir.rglob(TASK_FILE))
    return sorted((t for t in tasks if include_heldout or not t.heldout), key=lambda t: t.id)


def task_hash(task: CodingTask) -> str:
    """SHA-256 over every file in the task folder (task.toml, repo/, solution/,
    hidden_tests/, decoys/), by sorted relative path. Any byte change in any
    of them changes the hash; local clutter (HASH_IGNORED) does not."""
    digest = hashlib.sha256()
    for path in sorted(p for p in task.root.rglob("*") if p.is_file()):
        rel = path.relative_to(task.root)
        if HASH_IGNORED.intersection(rel.parts):
            continue
        digest.update(rel.as_posix().encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _validate(task: CodingTask) -> None:
    def fail(message: str) -> None:
        raise TaskError(f"task '{task.id}': {message}")

    if task.split not in SPLITS:
        fail(f"split must be one of {SPLITS}")
    # A task under a split folder (tasks/coding/dev/, tasks/coding/test/) must
    # declare that split, so a held-out task can't sit in dev/ by mistake.
    if task.root.parent.name in SPLITS and task.root.parent.name != task.split:
        fail(f"split '{task.split}' does not match its folder '{task.root.parent.name}/'")
    if task.id != task.root.name:
        fail(f"id must match its folder name '{task.root.name}'")
    if type(task.format) is not int or task.format not in FORMATS:  # excludes True == 1
        fail(f"format must be one of {FORMATS}")
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
        _check_editable_only(task, task.solution_dir, "solution/")
    if task.format == 1:
        for extra in (HIDDEN_TESTS_DIR, DECOYS_DIR):
            if (task.root / extra).exists():
                fail(f"{extra}/ needs format = 2")
        return

    # Format 2: hidden tests, decoys and sources are required.
    if not task.sources or not all(isinstance(s, str) and s for s in task.sources):
        fail("format 2 needs source = a non-empty list of bug-report URLs")
    # The verifier mounts the hidden tests over tests/ and reuses test_command,
    # so the command must target the tests/ folder, not individual files.
    if task.test_command[-1] != TESTS_DIR or any(c.startswith(f"{TESTS_DIR}/") for c in task.test_command):
        fail(f"format 2 test_command must end with the '{TESTS_DIR}' folder")
    hidden = task.root / HIDDEN_TESTS_DIR
    if not hidden.is_dir() or not any(p.is_file() for p in hidden.rglob("test_*.py")):
        fail(f"format 2 needs {HIDDEN_TESTS_DIR}/ with at least one test_*.py")
    if not task.solution_dir.is_dir():
        fail("format 2 needs a solution/")
    names = task.decoy_names()
    if not names:
        fail(f"format 2 needs at least one decoy in {DECOYS_DIR}/<name>/")
    for name in names:
        _check_editable_only(task, task.decoys_dir / name, f"{DECOYS_DIR}/{name}/", non_empty=True)


def _check_editable_only(task: CodingTask, folder: Path, label: str, *, non_empty: bool = False) -> None:
    files = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    if not files <= set(task.editable):
        raise TaskError(f"task '{task.id}': {label} may only contain editable files: "
                        f"{sorted(files - set(task.editable))}")
    if non_empty and not files:
        raise TaskError(f"task '{task.id}': {label} is empty")
