"""R1: plain BM25 on SciFact's development (train) queries, with a headroom diagnosis.

    venv/bin/python -m oarp.envs.retrieval.data                         # download once (verified)
    venv/bin/python experiments/retrieval_baseline/run_baseline.py      # free: no model calls

Model-free and deterministic. Uses only the train split; the test split is
never loaded. Writes:

- results/summary.json and results/analysis.md: aggregates only (public);
- results/private/per_query.jsonl: per-query ranks of the relevant documents,
  derived from the benchmark labels (gitignored, never published).

The headroom thresholds are triage guidelines decided before the run, not
automatic decisions: share of queries with no relevant document in the top 10
>= 20% suggests room for an adaptive policy, < 10% suggests a harder dataset,
in between needs a judgment call.
"""

import argparse
import json
import platform
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from oarp.core.trace import git_info  # noqa: E402
from oarp.envs.retrieval.bm25 import BM25Config, BM25Index, provenance  # noqa: E402
from oarp.envs.retrieval.data import DEFAULT_DATA_DIR, load_scifact  # noqa: E402
from oarp.envs.retrieval.metrics import (  # noqa: E402
    HEADROOM_BUCKETS,
    first_relevant_rank,
    headroom_bucket,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)

RESULTS = Path(__file__).resolve().parent / "results"
DEPTH = 100  # deepest cut-off reported; ranks beyond it are "beyond 100"
TRIAGE = {"suggests adaptive policy": 0.20, "suggests harder dataset": 0.10}


def evaluate(data_dir: Path, config: BM25Config = BM25Config()) -> tuple[dict, list[dict]]:
    split = load_scifact(data_dir, "train")
    started = time.perf_counter()
    index = BM25Index(split.corpus, config)
    index_s = time.perf_counter() - started

    rows = []
    started = time.perf_counter()
    for qid in sorted(split.queries):
        judged = split.qrels[qid]
        ranking = [doc for doc, _ in index.rank(split.queries[qid])]
        relevant_ranks = sorted(r for r, doc in enumerate(ranking, 1) if judged.get(doc, 0) > 0)
        first = first_relevant_rank(ranking, judged)
        rows.append({
            "query_id": qid, "relevant": sum(g > 0 for g in judged.values()),
            "relevant_ranks": relevant_ranks, "first_relevant_rank": first,
            "bucket": headroom_bucket(first),
            "ndcg@10": ndcg_at_k(ranking, judged, 10), "recall@10": recall_at_k(ranking, judged, 10),
            "recall@100": recall_at_k(ranking, judged, DEPTH), "mrr@10": mrr_at_k(ranking, judged, 10),
            "query_tokens_in_index": len(index.query_tokens(split.queries[qid])),
        })
    scoring_s = time.perf_counter() - started

    n = len(rows)
    buckets = Counter(r["bucket"] for r in rows)
    missed_top10 = sum(r["recall@10"] == 0 for r in rows) / n
    summary = {
        "study": "R1: plain BM25 on SciFact train queries (model-free baseline)",
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_info(REPO), "python": platform.python_version(), **provenance(config),
        "dataset": _archive_identity(data_dir),
        "split": "train",
        "counts": {
            "corpus_documents": len(split.corpus),
            "queries_in_file": split.total_queries_in_file,
            "judged_queries_in_split": n + sum(len(v) for v in split.skipped.values()),
            "evaluated_queries": n,
            "skipped": {reason: len(ids) for reason, ids in split.skipped.items()},
            "relevant_judgments": sum(r["relevant"] for r in rows),
            "zero_grade_judgments": sum(g == 0 for j in split.qrels.values() for g in j.values()),
            "queries_with_several_relevant": sum(r["relevant"] > 1 for r in rows),
            "queries_with_no_indexed_token": sum(r["query_tokens_in_index"] == 0 for r in rows),
        },
        "metrics_mean": {m: statistics.fmean(r[m] for r in rows)
                         for m in ("ndcg@10", "recall@10", "recall@100", "mrr@10")},
        "first_relevant_rank_buckets": {b: buckets.get(b, 0) for b in HEADROOM_BUCKETS},
        "queries_missing_some_relevant_in_top10": sum(0 < r["recall@10"] < 1 for r in rows),
        "share_no_relevant_in_top10": missed_top10,
        "triage_guideline": ("suggests adaptive policy" if missed_top10 >= TRIAGE["suggests adaptive policy"]
                             else "suggests harder dataset" if missed_top10 < TRIAGE["suggests harder dataset"]
                             else "judgment call"),
        "timing_s": {"index": round(index_s, 3), "scoring_all_queries": round(scoring_s, 3)},
    }
    return summary, rows


def _archive_identity(data_dir: Path) -> dict | None:
    path = data_dir / "scifact" / "ARCHIVE.json"
    return json.loads(path.read_text()) if path.is_file() else None


def report(summary: dict) -> str:
    c, m, b = summary["counts"], summary["metrics_mean"], summary["first_relevant_rank_buckets"]
    n = c["evaluated_queries"]
    lines = [
        "# R1: BM25 baseline on SciFact (train queries)", "",
        f"Model-free and deterministic. Corpus: {c['corpus_documents']} documents. "
        f"Queries in file: {c['queries_in_file']}; judged in the train split: {c['judged_queries_in_split']}; "
        f"evaluated: {n}; skipped: {c['skipped'] or 'none'}. The test split was not loaded.", "",
        "| nDCG@10 | Recall@10 | Recall@100 | MRR@10 |", "|---|---|---|---|",
        f"| {m['ndcg@10']:.4f} | {m['recall@10']:.4f} | {m['recall@100']:.4f} | {m['mrr@10']:.4f} |", "",
        "## Where the first relevant document lands", "",
        "| Bucket | Queries | Share |", "|---|---|---|",
        *[f"| {bucket} | {count} | {count / n:.1%} |" for bucket, count in b.items()], "",
        f"Queries with no relevant document in the top 10: {summary['share_no_relevant_in_top10']:.1%} "
        f"(triage guideline: {summary['triage_guideline']}; a guideline, not a decision).",
        f"Queries with several relevant documents: {c['queries_with_several_relevant']}; of all queries, "
        f"{summary['queries_missing_some_relevant_in_top10']} found some but not all relevant documents in the top 10.",
        f"Queries with no word in the index vocabulary: {c['queries_with_no_indexed_token']}.", "",
        f"BM25: {summary['bm25_config']}; bm25s {summary['bm25s']}, numpy {summary['numpy']}. "
        f"Timing: index {summary['timing_s']['index']} s, all queries {summary['timing_s']['scoring_all_queries']} s.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args(argv)

    summary, rows = evaluate(args.data_dir)
    (args.results / "private").mkdir(parents=True, exist_ok=True)
    (args.results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.results / "analysis.md").write_text(report(summary))
    with (args.results / "private" / "per_query.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(report(summary))
    return summary


if __name__ == "__main__":
    main()
