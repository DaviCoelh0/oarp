"""The agent's disposable working copy of a task repo, with the file-access rules.

- Read: any regular file inside the workspace (tests included).
- Write: whole-file replacement of the task's `editable` files only. No new
  files, no deletions. An allowlist, not a denylist: a denylist such as
  "anything but tests/" would still allow e.g. a new conftest.py that rewrites
  test outcomes.

Every model-supplied path is resolved (collapsing '..' and following symlinks)
and must stay inside the workspace. Violations raise ToolError, so the model
sees why and can correct itself.
"""

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from oarp.core.tools import ToolError
from oarp.envs.coding.sandbox import Mount
from oarp.envs.coding.task import TESTS_DIR, CodingTask

MAX_WRITE_BYTES = 100_000
CONTAINER_WORKDIR = "/work"


class Workspace:
    """`root` is the agent's copy of repo/. `pristine_tests` is a snapshot of the
    original tests taken at creation, outside `root`, so the agent's tools can't
    reach it; it is what the sandbox mounts over /work/tests. Docker therefore
    never mounts anything from the oarp repo itself."""

    def __init__(self, root: Path, pristine_tests: Path, task: CodingTask) -> None:
        self.root = root.resolve()
        self.pristine_tests = pristine_tests.resolve()
        self.task = task
        self._editable = {(self.root / rel).resolve() for rel in task.editable}

    @classmethod
    @contextmanager
    def create(
        cls, task: CodingTask, base_dir: Path | None = None, *, tests_from: Path | None = None
    ) -> Iterator["Workspace"]:
        """Copy the task into a fresh temp folder; delete it afterwards.

        `tests_from` replaces the snapshot mounted over tests/ (default: the
        public tests). The verifier passes the hidden tests here; they are
        copied next to the workspace, never into it, so no tool can reach them."""
        base = Path(tempfile.mkdtemp(prefix=f"oarp-{task.id}-", dir=base_dir))
        try:
            # mkdtemp creates the folder as 0700; the sandbox runs as another
            # user (nobody) and needs to read it. Copied contents keep their modes.
            base.chmod(0o755)
            root, pristine_tests = base / "work", base / "pristine_tests"
            shutil.copytree(task.repo_dir, root)
            shutil.copytree(tests_from or task.tests_dir, pristine_tests)
            yield cls(root, pristine_tests, task)
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def list_files(self) -> list[str]:
        return sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file())

    def read_text(self, path: str) -> str:
        target = self._resolve(path)
        if not target.is_file():
            raise ToolError(f"file not found: {path}")
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"not a UTF-8 text file: {path}") from None

    def write_text(self, path: str, content: str) -> None:
        target = self._resolve(path)
        if target not in self._editable:
            raise ToolError(
                f"not allowed to write {path}. Editable files: {', '.join(self.task.editable)}"
            )
        data = content.encode("utf-8")
        if len(data) > MAX_WRITE_BYTES:
            raise ToolError(f"content too large ({len(data)} bytes; limit {MAX_WRITE_BYTES})")
        target.write_bytes(data)

    def sandbox_mounts(self) -> list[Mount]:
        """Mounts for running the task's tests: the workspace at /work, with the
        pristine test snapshot mounted over /work/tests, so only original tests run."""
        return [
            Mount(self.root, CONTAINER_WORKDIR),
            Mount(self.pristine_tests, f"{CONTAINER_WORKDIR}/{TESTS_DIR}"),
        ]

    def _resolve(self, path: str) -> Path:
        if not path or Path(path).is_absolute():
            raise ToolError(f"path must be relative to the workspace: {path!r}")
        candidate = (self.root / path).resolve()
        if not candidate.is_relative_to(self.root):
            raise ToolError(f"path is outside the workspace: {path}")
        return candidate
