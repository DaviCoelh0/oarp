"""R2a free headroom analysis end to end: on the small BEIR-format fixture
(train split only, label details private), and, when the downloaded SciFact
data is present, against the R1 baseline and the numbers quoted in the R2
proposal. Skipped without the optional `retrieval` extra."""

import importlib.util
import json
import statistics
from pathlib import Path

import pytest

pytest.importorskip("bm25s")

from oarp.core.model import ScriptedModel  # noqa: E402
from oarp.core.types import ModelResponse  # noqa: E402
from oarp.envs.retrieval import rerank  # noqa: E402
from oarp.envs.retrieval.analysis import oracle_gate, oracle_rerank  # noqa: E402
from oarp.envs.retrieval.bm25 import BM25Index  # noqa: E402
from oarp.envs.retrieval.data import DEFAULT_DATA_DIR, load_scifact  # noqa: E402
from oarp.envs.retrieval.metrics import ndcg_at_k, recall_at_k  # noqa: E402
from oarp.envs.retrieval.strategies import combine  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "retrieval"
SPEC = importlib.util.spec_from_file_location(
    "run_analysis", REPO / "experiments" / "retrieval_rerank_headroom" / "run_analysis.py")
headroom = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(headroom)
R1_SUMMARY = REPO / "experiments" / "retrieval_baseline" / "results" / "summary.json"
requires_scifact = pytest.mark.skipif(not (DEFAULT_DATA_DIR / "scifact" / "corpus.jsonl").is_file(),
                                      reason="SciFact not downloaded (python -m oarp.envs.retrieval.data)")


def test_analysis_runs_on_train_only(tmp_path: Path, monkeypatch) -> None:
    loaded = []
    real = headroom.load_scifact
    monkeypatch.setattr(headroom, "load_scifact", lambda d, split, **kw: loaded.append((split, kw)) or real(d, split, **kw))
    summary = headroom.main(["--data-dir", str(FIXTURE), "--results", str(tmp_path)])
    assert loaded == [("train", {})]
    assert summary["queries"] == {"all": 3, "tune": 2, "check": 1} and summary["split"] == "train"
    assert set(summary["oracle_by_k"]) == {"10", "20", "30", "50", "100"}


def test_label_details_stay_private(tmp_path: Path) -> None:
    headroom.main(["--data-dir", str(FIXTURE), "--results", str(tmp_path)])
    rows = [json.loads(line) for line in (tmp_path / "private" / "per_query.jsonl").read_text().splitlines()]
    assert [r["query_id"] for r in rows] == ["q1", "q2", "q3"]
    public = (tmp_path / "summary.json").read_text() + (tmp_path / "analysis.md").read_text()
    for qid in ("q1", "q2", "q3"):
        assert f'"{qid}"' not in public and f"{qid} " not in public


def test_oracle_gate_through_the_combiner_matches_the_analysis(tmp_path: Path) -> None:
    summary = headroom.main(["--data-dir", str(FIXTURE), "--results", str(tmp_path)])
    split = load_scifact(FIXTURE, "train")
    index = BM25Index(split.corpus)
    bm25 = {q: [d for d, _ in index.rank(t)] for q, t in split.queries.items()}
    k = summary["rerank_k"]
    final = combine(bm25, {q: oracle_rerank(bm25[q], split.qrels[q], k) for q in bm25}, oracle_gate(bm25, split.qrels))
    mean = statistics.fmean(ndcg_at_k(final[q], split.qrels[q], 10) for q in bm25)
    assert mean == pytest.approx(summary["oracle_gate"]["all"]["oracle_gate_oracle_reranker"]["ndcg@10"])


@requires_scifact
def test_identity_reranker_reproduces_r1() -> None:
    split = load_scifact(DEFAULT_DATA_DIR, "train")
    index = BM25Index(split.corpus)
    r1 = json.loads(R1_SUMMARY.read_text())["metrics_mean"]
    ndcg, rec = [], []
    for qid in sorted(split.queries):
        bm25 = [d for d, _ in index.rank(split.queries[qid])]
        model = ScriptedModel([ModelResponse(text="RANKING: " + ", ".join(str(i) for i in range(1, 21)))])
        record = rerank.rerank(model, qid, split.queries[qid], bm25, split.corpus)
        assert record.ranking == bm25 and record.status == "valid"
        ndcg.append(ndcg_at_k(record.ranking, split.qrels[qid], 10))
        rec.append(recall_at_k(record.ranking, split.qrels[qid], 100))
    assert statistics.fmean(ndcg) == pytest.approx(r1["ndcg@10"], abs=1e-12)
    assert statistics.fmean(rec) == pytest.approx(r1["recall@100"], abs=1e-12)


@requires_scifact
def test_oracle_table_matches_the_proposal(tmp_path: Path) -> None:
    summary = headroom.main(["--results", str(tmp_path)])
    quoted = {"10": (0.802, 0.791, 0.787), "20": (0.854, 0.845, 0.842), "30": (0.873, 0.863, 0.861),
              "50": (0.883, 0.876, 0.874), "100": (0.912, 0.908, 0.906)}
    for k, (cover, ndcg, rec) in quoted.items():
        row = summary["oracle_by_k"][k]["all"]
        assert round(row["first_relevant_in_top_k"], 3) == cover
        assert round(row["ndcg@10"], 3) == ndcg and round(row["recall@10"], 3) == rec
