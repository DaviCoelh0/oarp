import os
from pathlib import Path

import pytest
from conftest import TASKS_DIR

from oarp.envs.coding.task import TaskError, discover_tasks, load_task

VALID_TOML = """\
id = "t1"
split = "dev"
prompt = "Fix it."
editable = ["src/m.py"]
test_command = ["python", "-m", "pytest", "tests"]
timeout_s = 10
"""


def make_task(tmp_path: Path, toml: str = VALID_TOML) -> Path:
    root = tmp_path / "t1"
    (root / "repo" / "src").mkdir(parents=True)
    (root / "repo" / "tests").mkdir()
    (root / "repo" / "src" / "m.py").write_text("x = 1\n")
    (root / "repo" / "tests" / "test_m.py").write_text("def test_x(): pass\n")
    (root / "task.toml").write_text(toml)
    return root


def test_fixture_task_loads() -> None:
    task = load_task(TASKS_DIR / "coding" / "dev" / "off_by_one_001")
    assert task.id == "off_by_one_001"
    assert task.split == "dev"
    assert task.editable == ("src/stats.py",)
    assert task.test_command[:3] == ("python", "-m", "pytest")


def test_discover_finds_all_tasks() -> None:
    assert "off_by_one_001" in [t.id for t in discover_tasks(TASKS_DIR)]


def test_minimal_valid_task(tmp_path: Path) -> None:
    assert load_task(make_task(tmp_path)).timeout_s == 10.0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (('split = "dev"', 'split = "train"'), "split must be one of"),
        (('id = "t1"', 'id = "other"'), "must match its folder name"),
        (('["src/m.py"]', '["tests/test_m.py"]'), "tests can never be editable"),
        (('["src/m.py"]', '["../outside.py"]'), "without '..'"),
        (('["src/m.py"]', '["/etc/passwd"]'), "without '..'"),
        (('["src/m.py"]', '["src/missing.py"]'), "does not exist"),
        (('["src/m.py"]', "[]"), "non-empty list"),
        (("timeout_s = 10", "timeout_s = 0"), "must be positive"),
        (('prompt = "Fix it."\n', ""), "'prompt' is missing"),
    ],
)
def test_invalid_definitions_rejected(tmp_path: Path, change: tuple[str, str], message: str) -> None:
    root = make_task(tmp_path, VALID_TOML.replace(*change))
    with pytest.raises(TaskError, match=message):
        load_task(root)


def test_symlinks_rejected(tmp_path: Path) -> None:
    root = make_task(tmp_path)
    os.symlink("/etc/hosts", root / "repo" / "src" / "link.py")
    with pytest.raises(TaskError, match="symlinks"):
        load_task(root)


def test_solution_may_only_contain_editable_files(tmp_path: Path) -> None:
    root = make_task(tmp_path)
    (root / "solution" / "tests").mkdir(parents=True)
    (root / "solution" / "tests" / "test_m.py").write_text("")
    with pytest.raises(TaskError, match="solution/ may only contain editable files"):
        load_task(root)
