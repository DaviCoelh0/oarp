"""R1 baseline script end to end on the small BEIR-format dataset: train split
only, aggregates public, per-query label-derived details private."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("bm25s")

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "retrieval"
SPEC = importlib.util.spec_from_file_location("run_baseline", REPO / "experiments" / "retrieval_baseline" / "run_baseline.py")
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


def test_baseline_runs_on_train_only(tmp_path: Path, monkeypatch) -> None:
    loaded = []
    real = baseline.load_scifact
    monkeypatch.setattr(baseline, "load_scifact", lambda d, split, **kw: loaded.append((split, kw)) or real(d, split, **kw))
    summary = baseline.main(["--data-dir", str(FIXTURE), "--results", str(tmp_path)])

    assert loaded == [("train", {})]  # never the test split, never the held-out opt-in
    c = summary["counts"]
    assert (c["corpus_documents"], c["queries_in_file"], c["judged_queries_in_split"], c["evaluated_queries"]) == (5, 5, 5, 3)
    assert c["skipped"] == {"not in queries file": 1, "no relevant document": 1}
    assert c["zero_grade_judgments"] == 1 and c["queries_with_several_relevant"] == 1
    assert set(summary["first_relevant_rank_buckets"]) == {"rank 1", "ranks 2-10", "ranks 11-100", "beyond 100"}
    assert sum(summary["first_relevant_rank_buckets"].values()) == 3
    assert summary["bm25s"] == "0.3.13" and summary["split"] == "train"


def test_label_details_stay_private(tmp_path: Path) -> None:
    baseline.main(["--data-dir", str(FIXTURE), "--results", str(tmp_path)])
    rows = [json.loads(line) for line in (tmp_path / "private" / "per_query.jsonl").read_text().splitlines()]
    assert [r["query_id"] for r in rows] == ["q1", "q2", "q3"]
    public = (tmp_path / "summary.json").read_text() + (tmp_path / "analysis.md").read_text()
    for qid in ("q1", "q2", "q3"):
        assert f'"{qid}"' not in public and f"{qid} " not in public


def test_results_are_deterministic(tmp_path: Path) -> None:
    first = baseline.evaluate(FIXTURE)[1]
    second = baseline.evaluate(FIXTURE)[1]
    assert first == second
