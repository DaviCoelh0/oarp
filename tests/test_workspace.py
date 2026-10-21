import os
from pathlib import Path

import pytest
from conftest import TASKS_DIR

from oarp.core.tools import ToolError
from oarp.envs.coding.task import load_task
from oarp.envs.coding.workspace import MAX_WRITE_BYTES, Workspace

TASK = load_task(TASKS_DIR / "coding" / "dev" / "off_by_one_001")
ORIGINAL = (TASK.repo_dir / "src" / "stats.py").read_text()


@pytest.fixture
def ws():
    with Workspace.create(TASK) as workspace:
        yield workspace


def test_workspace_is_an_independent_copy(ws: Workspace) -> None:
    assert ws.root != TASK.repo_dir and not ws.root.is_relative_to(TASK.root)
    assert ws.list_files() == ["src/stats.py", "tests/test_stats.py"]
    ws.write_text("src/stats.py", "# changed\n")
    assert ws.read_text("src/stats.py") == "# changed\n"
    assert (TASK.repo_dir / "src" / "stats.py").read_text() == ORIGINAL  # original untouched


def test_workspace_is_deleted_afterwards() -> None:
    with Workspace.create(TASK) as workspace:
        root = workspace.root
        assert root.exists()
    assert not root.exists()


def test_workspace_deleted_even_on_error() -> None:
    with pytest.raises(RuntimeError):
        with Workspace.create(TASK) as workspace:
            root = workspace.root
            raise RuntimeError("boom")
    assert not root.exists()


def test_can_read_source_and_tests(ws: Workspace) -> None:
    assert "def moving_average" in ws.read_text("src/stats.py")
    assert "def test_basic_window" in ws.read_text("tests/test_stats.py")


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("../../.env", "outside the workspace"),
        ("src/../../outside.txt", "outside the workspace"),
        ("/etc/passwd", "must be relative"),
        ("", "must be relative"),
        ("src/missing.py", "file not found"),
        ("src", "file not found"),  # a directory, not a file
    ],
)
def test_bad_read_paths_rejected(ws: Workspace, path: str, message: str) -> None:
    with pytest.raises(ToolError, match=message):
        ws.read_text(path)


@pytest.mark.parametrize(
    "path",
    [
        "tests/test_stats.py",  # tests are never editable
        "src/../tests/test_stats.py",  # sneaking past the allowlist via '..'
        "conftest.py",  # new file that pytest would load automatically
        "tests/conftest.py",
        "src/new_module.py",  # no new files at all
    ],
)
def test_writes_outside_allowlist_rejected(ws: Workspace, path: str) -> None:
    with pytest.raises(ToolError, match="not allowed to write"):
        ws.write_text(path, "def pytest_runtest_makereport(item, call): ...\n")
    assert ws.list_files() == ["src/stats.py", "tests/test_stats.py"]


@pytest.mark.parametrize("path", ["src/stats.py", "./src/stats.py", "src//stats.py", "src/../src/stats.py"])
def test_equivalent_spellings_of_editable_path_allowed(ws: Workspace, path: str) -> None:
    ws.write_text(path, "# fixed\n")
    assert ws.read_text("src/stats.py") == "# fixed\n"


def test_symlink_escape_rejected(ws: Workspace, tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("sk-ant-FAKE")
    os.symlink(secret, ws.root / "src" / "link.py")  # simulates a link appearing in the workspace
    with pytest.raises(ToolError, match="outside the workspace"):
        ws.read_text("src/link.py")


def test_oversized_write_rejected(ws: Workspace) -> None:
    with pytest.raises(ToolError, match="too large"):
        ws.write_text("src/stats.py", "x" * (MAX_WRITE_BYTES + 1))
    assert ws.read_text("src/stats.py") == ORIGINAL


def test_non_utf8_file_rejected(ws: Workspace) -> None:
    (ws.root / "src" / "blob.bin").write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ToolError, match="not a UTF-8"):
        ws.read_text("src/blob.bin")


def test_sandbox_mounts_use_pristine_test_snapshot(ws: Workspace) -> None:
    work, tests = ws.sandbox_mounts()
    assert (work.source, work.target) == (ws.root, "/work")
    assert (tests.source, tests.target) == (ws.pristine_tests, "/work/tests")
    # The snapshot sits outside the agent's root, so its tools can't reach it ...
    assert not ws.pristine_tests.is_relative_to(ws.root)
    # ... and it is byte-identical to the original tests.
    for original in TASK.tests_dir.rglob("*.py"):
        assert (ws.pristine_tests / original.relative_to(TASK.tests_dir)).read_bytes() == original.read_bytes()
    # Docker never mounts anything from the oarp repo.
    assert not any(m.source.is_relative_to(TASKS_DIR.parent) for m in ws.sandbox_mounts())


def test_test_snapshot_cannot_be_reached_via_tools(ws: Workspace) -> None:
    with pytest.raises(ToolError, match="outside the workspace"):
        ws.read_text("../pristine_tests/test_stats.py")
