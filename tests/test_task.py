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


# --- format 2, splits, held-out, task hash -------------------------------------------------

from conftest import FIXTURES_DIR  # noqa: E402

from oarp.envs.coding.task import task_hash  # noqa: E402

F2 = FIXTURES_DIR / "count_words_f2"


def copy_f2(tmp_path: Path, parent: str = "anywhere") -> Path:
    import shutil

    root = tmp_path / parent / "count_words_f2"
    shutil.copytree(F2, root)
    return root


def test_format2_fixture_loads() -> None:
    task = load_task(F2)
    assert (task.format, task.sources) == (2, ("https://example.org/fixture-only",))
    assert task.hidden_tests_dir == F2 / "hidden_tests"
    assert task.decoy_names() == ["drop_empty_pieces", "strip_then_split"]


def test_format1_tasks_have_no_hidden_tests() -> None:
    task = load_task(TASKS_DIR / "coding" / "dev" / "off_by_one_001")
    assert task.format == 1 and task.hidden_tests_dir is None and task.decoy_names() == []


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda r: (r / "task.toml").write_text((r / "task.toml").read_text().replace(
            'source = ["https://example.org/fixture-only"]\n', "")), "needs source"),
        (lambda r: (r / "task.toml").write_text((r / "task.toml").read_text().replace(
            'source = ["https://example.org/fixture-only"]', 'source = "https://x"')), "must be a list"),
        (lambda r: (r / "task.toml").write_text((r / "task.toml").read_text().replace(
            "format = 2", "format = 3")), "format must be one of"),
        (lambda r: (r / "task.toml").write_text((r / "task.toml").read_text().replace(
            "format = 2", "format = true")), "format must be one of"),
        (lambda r: (r / "task.toml").write_text((r / "task.toml").read_text().replace(
            '"tests"]', '"tests/test_words.py"]')), "must end with the 'tests' folder"),
        (lambda r: __import__("shutil").rmtree(r / "hidden_tests"), "needs hidden_tests/"),
        (lambda r: __import__("shutil").rmtree(r / "decoys"), "at least one decoy"),
        (lambda r: __import__("shutil").rmtree(r / "solution"), "needs a solution/"),
        (lambda r: (r / "decoys" / "strip_then_split" / "notes.txt").write_text("x"),
         "may only contain editable files"),
    ],
)
def test_invalid_format2_definitions_rejected(tmp_path: Path, edit, message: str) -> None:
    root = copy_f2(tmp_path)
    edit(root)
    with pytest.raises(TaskError, match=message):
        load_task(root)


@pytest.mark.parametrize("extra", ["hidden_tests", "decoys"])
def test_format1_rejects_format2_folders(tmp_path: Path, extra: str) -> None:
    root = make_task(tmp_path)
    (root / extra).mkdir()
    with pytest.raises(TaskError, match="needs format = 2"):
        load_task(root)


@pytest.mark.parametrize(("folder", "split"), [("dev", "test"), ("test", "dev")])
def test_split_must_match_its_folder(tmp_path: Path, folder: str, split: str) -> None:
    root = make_task(tmp_path / folder, VALID_TOML.replace('split = "dev"', f'split = "{split}"'))
    with pytest.raises(TaskError, match="does not match its folder"):
        load_task(root)


def test_heldout_tasks_are_skipped_unless_requested(tmp_path: Path) -> None:
    make_task(tmp_path / "dev")
    heldout = tmp_path / "test"
    make_task(heldout, VALID_TOML.replace('split = "dev"', 'split = "test"').replace('id = "t1"', 'id = "t2"'))
    (heldout / "t1").rename(heldout / "t2")  # make_task always names the folder t1
    assert [t.id for t in discover_tasks(tmp_path)] == ["t1"]
    assert [t.id for t in discover_tasks(tmp_path, include_heldout=True)] == ["t1", "t2"]


def test_task_hash_is_stable() -> None:
    assert task_hash(load_task(F2)) == task_hash(load_task(F2))


@pytest.mark.parametrize("rel", [
    "task.toml", "repo/src/words.py", "repo/tests/test_words.py", "hidden_tests/test_words_hidden.py",
    "solution/src/words.py", "decoys/strip_then_split/src/words.py",
])
def test_task_hash_changes_with_any_byte(tmp_path: Path, rel: str) -> None:
    root = copy_f2(tmp_path)
    before = task_hash(load_task(root))
    with (root / rel).open("a") as f:
        f.write(" ")
    assert task_hash(load_task(root)) != before


def test_task_hash_ignores_local_clutter(tmp_path: Path) -> None:
    root = copy_f2(tmp_path)
    before = task_hash(load_task(root))
    (root / "repo" / "src" / "__pycache__").mkdir()
    (root / "repo" / "src" / "__pycache__" / "words.cpython-311.pyc").write_bytes(b"\x00")
    (root / ".DS_Store").write_bytes(b"\x00")
    assert task_hash(load_task(root)) == before
