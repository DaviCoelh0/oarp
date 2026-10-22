"""The validity checks themselves must catch broken tasks (a check that never
fails proves nothing). Each test breaks a copy of the fixture in one way."""

import shutil
from pathlib import Path

import pytest
from conftest import FIXTURES_DIR, requires_docker

from oarp.envs.coding.task import load_task
from oarp.envs.coding.validity import behaviour_problems, static_problems

F2 = FIXTURES_DIR / "count_words_f2"
ORIGINAL = (F2 / "repo" / "src" / "words.py").read_text()
SOLUTION = (F2 / "solution" / "src" / "words.py").read_text()


def broken_copy(tmp_path: Path) -> Path:
    root = tmp_path / "count_words_f2"
    shutil.copytree(F2, root)
    return root


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        (lambda r: (r / "repo" / "src" / "notes.py").write_text(SOLUTION), "identical to repo/src/notes.py"),
        (lambda r: (r / "decoys" / "strip_then_split" / "src" / "words.py").write_text(ORIGINAL),
         "decoy strip_then_split file src/words.py is identical to repo/src/words.py"),
        (lambda r: (r / "repo" / "src" / "words.py").write_text(ORIGINAL + "# see the solution\n"),
         "repo/src/words.py mentions 'solution'"),
        (lambda r: (r / "task.toml").write_text((r / "task.toml").read_text().replace(
            "Word counts", "Hidden word counts")), "prompt mentions 'hidden'"),
    ],
)
def test_static_check_catches_leaks(tmp_path: Path, edit, expected: str) -> None:
    root = broken_copy(tmp_path)
    edit(root)
    assert any(expected in p for p in static_problems(load_task(root)))


@pytest.mark.docker
@requires_docker
def test_behaviour_check_catches_a_decoy_that_passes(tmp_path: Path) -> None:
    root = broken_copy(tmp_path)
    (root / "decoys" / "strip_then_split" / "src" / "words.py").write_text(SOLUTION.replace("split()", "split( )"))
    # The remaining decoy fails only hidden tests, so both problems are reported.
    assert behaviour_problems(load_task(root)) == [
        "decoy strip_then_split: passed, expected tests_failed", "no decoy fails the public tests"]


@pytest.mark.docker
@requires_docker
def test_behaviour_check_needs_a_publicly_failing_decoy(tmp_path: Path) -> None:
    root = broken_copy(tmp_path)
    shutil.rmtree(root / "decoys" / "strip_then_split")  # left: the decoy that only fails hidden tests
    assert behaviour_problems(load_task(root)) == ["no decoy fails the public tests"]


@pytest.mark.docker
@requires_docker
def test_behaviour_check_catches_a_reference_failing_hidden_tests(tmp_path: Path) -> None:
    root = broken_copy(tmp_path)
    with (root / "hidden_tests" / "test_words_hidden.py").open("a") as f:
        f.write("\n\ndef test_impossible():\n    assert count_words('a') == 2\n")
    problems = behaviour_problems(load_task(root))
    assert problems == ["reference: tests_failed (public passed, hidden tests_failed)"]
