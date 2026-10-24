"""Shared study machinery: budget guard, preflight refusals and seeded ordering."""

import json
import subprocess
from pathlib import Path

import pytest

from oarp.envs.retrieval import study

PINNED = "a" * 64


def test_budget_guard_stops_before_the_cap() -> None:
    guard = study.BudgetGuard(0.01, 0.0035)
    calls = 0
    while guard.can_call():
        guard.add(0.003)
        calls += 1
    assert calls == 3 and guard.spent == pytest.approx(0.009) and guard.spent <= 0.01
    assert "would exceed" in guard.stop_reason()


def test_missing_cost_counts_at_worst_case() -> None:
    guard = study.BudgetGuard(0.01, 0.0035)
    calls = 0
    while guard.can_call():
        guard.add(None)
        calls += 1
    assert calls == 2 and guard.spent == pytest.approx(0.007)


def test_a_call_above_the_bound_stops_everything() -> None:
    guard = study.BudgetGuard(1.0, 0.0035)
    guard.add(0.005)
    assert not guard.can_call() and guard.calls_above_worst_case == 1 and "unsafe" in guard.stop_reason()


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args],
                   check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.py").write_text("x = 1\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


@pytest.fixture
def data(tmp_path: Path) -> Path:
    (tmp_path / "data" / "scifact").mkdir(parents=True)
    (tmp_path / "data" / "scifact" / "ARCHIVE.json").write_text(json.dumps({"sha256": PINNED}))
    return tmp_path / "data"


def test_preflight_accepts_a_clean_state(repo: Path, data: Path, tmp_path: Path) -> None:
    assert study.preflight(repo, tmp_path / "out", data, PINNED, paid=True)["git"]["dirty"] is False


def test_preflight_refusals(repo: Path, data: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    (repo / "src" / "a.py").write_text("x = 2\n")
    with pytest.raises(SystemExit, match="uncommitted"):
        study.preflight(repo, out, data, PINNED, paid=True)
    _git(repo, "commit", "-q", "-am", "edit")
    (repo / "src" / "new.py").write_text("")
    with pytest.raises(SystemExit, match="untracked"):
        study.preflight(repo, out, data, PINNED, paid=True)
    (repo / "src" / "new.py").unlink()
    out.mkdir()
    (out / "manifest.json").write_text("{}")
    with pytest.raises(SystemExit, match="not empty"):
        study.preflight(repo, out, data, PINNED, paid=True)
    (out / "manifest.json").unlink()
    with pytest.raises(SystemExit, match="SHA-256"):
        study.preflight(repo, out, data, "b" * 64, paid=True)
    assert study.preflight(repo, out, data, "b" * 64, paid=False)["git"] is not None  # dry runs only record


def test_hash_order_and_pairs_are_deterministic() -> None:
    ids = [f"q{i}" for i in range(50)]
    order = study.hash_order("seed:", ids)
    assert sorted(order) == sorted(ids) and order == study.hash_order("seed:", list(reversed(ids)))
    assert order != study.hash_order("other:", ids)
    pairs = {study.first_of_pair("seed:", q, ("a", "b")) for q in ids}
    assert pairs == {("a", "b"), ("b", "a")}
    assert study.first_of_pair("seed:", "q1", ("a", "b")) == study.first_of_pair("seed:", "q1", ("a", "b"))


def test_records_and_manifest_are_written(tmp_path: Path) -> None:
    study.append_record(tmp_path / "private" / "calls.jsonl", {"a": 1})
    study.append_record(tmp_path / "private" / "calls.jsonl", {"a": 2})
    study.write_manifest(tmp_path, {"n": 2})
    lines = (tmp_path / "private" / "calls.jsonl").read_text().splitlines()
    assert [json.loads(line)["a"] for line in lines] == [1, 2]
    assert json.loads((tmp_path / "manifest.json").read_text()) == {"n": 2}
