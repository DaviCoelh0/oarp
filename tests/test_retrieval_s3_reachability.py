"""S3a free reachability / RM3 analysis end to end on the small BEIR-format
fixture: train split and tune half only, label details private. Skipped
without the optional `retrieval` extra."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("bm25s")

from oarp.envs.retrieval.data import dev_partition, load_scifact  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "retrieval"
SPEC = importlib.util.spec_from_file_location(
    "s3_reachability", REPO / "experiments" / "retrieval_s3_reachability" / "run_analysis.py")
reach = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reach)


def test_runs_on_the_tune_half_of_train_only(tmp_path: Path, monkeypatch) -> None:
    loaded = []
    real = reach.load_scifact
    monkeypatch.setattr(reach, "load_scifact", lambda d, split, **kw: loaded.append((split, kw)) or real(d, split, **kw))
    summary = reach.main(["--data-dir", str(FIXTURE), "--results", str(tmp_path), "--depth", "1"])
    assert loaded == [("train", {})]
    tune = dev_partition(sorted(load_scifact(FIXTURE, "train").queries))["tune"]
    rows = [json.loads(line) for line in (tmp_path / "private" / "per_query.jsonl").read_text().splitlines()]
    assert [r["query_id"] for r in rows] == tune and summary["queries"] == len(tune)
    assert summary["strata"]["within_depth"] + summary["strata"]["beyond_depth"] == len(tune)
    for row in rows:
        if row["beyond"]:
            assert row["title_own_rank"] >= 1 and row["title_any_relevant_rank"] <= row["title_own_rank"]
    for value in summary["title_reachability_beyond_depth"].values():
        assert value is None or 0 <= value <= 1


def test_label_details_stay_private(tmp_path: Path) -> None:
    reach.main(["--data-dir", str(FIXTURE), "--results", str(tmp_path), "--depth", "1"])
    public = (tmp_path / "summary.json").read_text() + (tmp_path / "analysis.md").read_text()
    for qid in ("q1", "q2", "q3"):
        assert f'"{qid}"' not in public and f"{qid} " not in public
