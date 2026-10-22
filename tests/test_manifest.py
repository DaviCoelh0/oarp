"""Frozen task-set versions (invariants 16-17): a manifest of task hashes,
checked before any experiment batch, plus the `oarp tasks` CLI."""

import json
import shutil
from pathlib import Path

import pytest
from conftest import FIXTURES_DIR, TASKS_DIR

from oarp.cli import EXIT_FAILURE, EXIT_INVALID, EXIT_SUCCESS, main
from oarp.envs.coding.manifest import check_manifest, write_manifest
from oarp.envs.coding.task import discover_tasks


@pytest.fixture
def tasks_dir(tmp_path: Path) -> Path:
    root = tmp_path / "tasks"
    shutil.copytree(TASKS_DIR / "coding" / "dev", root / "coding" / "dev")
    shutil.copytree(FIXTURES_DIR / "count_words_f2", root / "coding" / "dev" / "count_words_f2")
    return root


def test_unchanged_tasks_match(tasks_dir: Path, tmp_path: Path) -> None:
    manifest = write_manifest(discover_tasks(tasks_dir), tmp_path / "m.json")
    assert len(manifest["tasks"]) == 6 and len(manifest["set_hash"]) == 64
    assert check_manifest(tmp_path / "m.json", tasks_dir) == []


@pytest.mark.parametrize("rel", ["repo/src/words.py", "hidden_tests/test_words_hidden.py",
                                 "decoys/drop_empty_pieces/src/words.py", "task.toml"])
def test_one_byte_change_is_named(tasks_dir: Path, tmp_path: Path, rel: str) -> None:
    write_manifest(discover_tasks(tasks_dir), tmp_path / "m.json")
    with (tasks_dir / "coding" / "dev" / "count_words_f2" / rel).open("a") as f:
        f.write("#")
    assert check_manifest(tmp_path / "m.json", tasks_dir) == [
        "count_words_f2: changed since the manifest was written"]


def test_missing_task_and_edited_manifest_are_reported(tasks_dir: Path, tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    write_manifest(discover_tasks(tasks_dir), path)
    shutil.rmtree(tasks_dir / "coding" / "dev" / "lru_cache_005")
    manifest = json.loads(path.read_text())
    manifest["tasks"][0]["hash"] = "0" * 64
    path.write_text(json.dumps(manifest))
    problems = check_manifest(path, tasks_dir)
    assert f"lru_cache_005: missing under {tasks_dir}" in problems
    assert "the manifest file itself was edited (set_hash mismatch)" in problems


def test_empty_task_set_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        write_manifest([], tmp_path / "m.json")


def test_cli_freeze_and_check(tasks_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    out = tmp_path / "m2_dev.json"
    assert main(["tasks", "freeze", "--split", "dev", "--ids", "count_words_f2", "off_by_one_001",
                 "--tasks-dir", str(tasks_dir), "--output", str(out)]) == EXIT_SUCCESS
    assert [t["id"] for t in json.loads(out.read_text())["tasks"]] == ["count_words_f2", "off_by_one_001"]
    assert main(["tasks", "check", str(out), "--tasks-dir", str(tasks_dir)]) == EXIT_SUCCESS
    (tasks_dir / "coding" / "dev" / "off_by_one_001" / "task.toml").write_text(
        (tasks_dir / "coding" / "dev" / "off_by_one_001" / "task.toml").read_text() + "\n")
    assert main(["tasks", "check", str(out), "--tasks-dir", str(tasks_dir)]) == EXIT_FAILURE
    assert "MISMATCH off_by_one_001: changed" in capsys.readouterr().out


def test_cli_freeze_rejects_unknown_ids(tasks_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(["tasks", "freeze", "--split", "dev", "--ids", "nope", "--tasks-dir", str(tasks_dir),
                 "--output", str(tmp_path / "m.json")]) == EXIT_INVALID
    assert "no dev task(s) named ['nope']" in capsys.readouterr().err
    assert not (tmp_path / "m.json").exists()
