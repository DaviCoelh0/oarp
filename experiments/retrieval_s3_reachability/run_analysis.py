"""S3a: free reachability analysis and the RM3 baseline on SciFact's tune half.

    venv/bin/python experiments/retrieval_s3_reachability/run_analysis.py   # free: no model calls

Model-free and deterministic. Uses the tune half of the development (train)
queries only: the check half stays untouched for the S3 line, and the test
split is never loaded. Dev labels are used for analysis only. Reports:

1. strata by BM25's first-relevant rank (within / beyond the first search's
   `depth` = 20 candidates), as used for the stratified S3 sample;
2. title reachability (an oracle): for each beyond-20 query, search BM25 with
   each relevant document's own title and record that document's rank and the
   first rank of any relevant document. The title is part of the indexed
   text, so this is close to a known-item search: an optimistic, indicative
   ceiling for "a good reformulation with BM25", not a strict bound in either
   direction;
3. RM3 (fixed defaults, not tuned) against BM25 on all tune queries
   (nDCG@10, Recall@20, Recall@100, paired bootstrap CIs) and per stratum:
   beyond-20 queries rescued into RM3's top 20 or top 10 (what one extra
   10-result search could add), and within-20 queries whose relevant documents
   all leave RM3's top 20 (harm).

Writes results/summary.json and results/analysis.md (aggregates only) and
results/private/per_query.jsonl (label-derived, gitignored).
"""

import argparse
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from oarp.core.trace import git_info  # noqa: E402
from oarp.envs.retrieval.bm25 import BM25Config, BM25Index, provenance  # noqa: E402
from oarp.envs.retrieval.data import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEV_PARTITION_SEED,
    dev_partition,
    load_scifact,
    load_titles,
)
from oarp.envs.retrieval.metrics import first_relevant_rank, ndcg_at_k, recall_at_k, relevant  # noqa: E402
from oarp.envs.retrieval.rm3 import RM3Config, rm3_rank  # noqa: E402
from oarp.envs.retrieval.rm3 import provenance as rm3_provenance  # noqa: E402
from oarp.envs.retrieval.stats import paired_bootstrap_ci  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
DEPTH = 20  # the forced first search's candidates (AgentConfig.first_k)
SEARCH_K = 10  # results of one model search (AgentConfig.search_k)
METRICS = {"ndcg@10": lambda r, j: ndcg_at_k(r, j, 10), "recall@20": lambda r, j: recall_at_k(r, j, 20),
           "recall@100": lambda r, j: recall_at_k(r, j, 100)}


def within(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def share(flags: list[bool]) -> float | None:
    return statistics.fmean(flags) if flags else None


def evaluate(data_dir: Path, depth: int = DEPTH, config: BM25Config = BM25Config(),
             rm3_config: RM3Config = RM3Config()) -> tuple[dict, list[dict]]:
    split = load_scifact(data_dir, "train")
    titles = load_titles(data_dir)
    index = BM25Index(split.corpus, config)
    tune = dev_partition(sorted(split.queries))["tune"]
    qrels = split.qrels

    rows = []
    for q in tune:
        judged = qrels[q]
        bm25 = [d for d, _ in index.rank(split.queries[q])]
        rm3 = [d for d, _ in rm3_rank(index, split.corpus, split.queries[q], rm3_config)]
        row = {"query_id": q, "bm25_first_rank": first_relevant_rank(bm25, judged),
               "rm3_first_rank": first_relevant_rank(rm3, judged),
               **{f"bm25_{m}": f(bm25, judged) for m, f in METRICS.items()},
               **{f"rm3_{m}": f(rm3, judged) for m, f in METRICS.items()}}
        row["beyond"] = not within(row["bm25_first_rank"], depth)
        if row["beyond"]:
            own, anyrel = [], []
            for doc_id in relevant(judged):
                ranking = [d for d, _ in index.rank(titles[doc_id])]
                own.append(ranking.index(doc_id) + 1)
                anyrel.append(first_relevant_rank(ranking, judged))
            row["title_own_rank"] = min(own)
            row["title_any_relevant_rank"] = min(r for r in anyrel if r is not None)
        rows.append(row)

    beyond = [r for r in rows if r["beyond"]]
    inside = [r for r in rows if not r["beyond"]]
    buckets = {"within": 0, "beyond_to_100": 0, "beyond_to_1000": 0, "beyond_1000": 0}
    for r in rows:
        rank = r["bm25_first_rank"]
        key = ("within" if within(rank, depth) else "beyond_to_100" if within(rank, 100)
               else "beyond_to_1000" if within(rank, 1000) else "beyond_1000")
        buckets[key] += 1

    title = {f"{kind}_within_{k}": share([within(r[f"title_{kind}_rank"], k) for r in beyond])
             for kind in ("own", "any_relevant") for k in (SEARCH_K, depth)}
    rm3_vs_bm25 = {m: {"bm25": statistics.fmean(r[f"bm25_{m}"] for r in rows),
                       "rm3": statistics.fmean(r[f"rm3_{m}"] for r in rows),
                       "rm3_minus_bm25": paired_bootstrap_ci([r[f"rm3_{m}"] - r[f"bm25_{m}"] for r in rows])}
                   for m in METRICS}
    rm3_strata = {
        "beyond_rescued_within_search_k": share([within(r["rm3_first_rank"], SEARCH_K) for r in beyond]),
        "beyond_rescued_within_depth": share([within(r["rm3_first_rank"], depth) for r in beyond]),
        "inside_lost_from_depth": share([not within(r["rm3_first_rank"], depth) for r in inside]),
        "inside_ndcg@10_rm3_minus_bm25": (statistics.fmean(r["rm3_ndcg@10"] - r["bm25_ndcg@10"] for r in inside)
                                          if inside else None),
        "beyond_ndcg@10_rm3_minus_bm25": (statistics.fmean(r["rm3_ndcg@10"] - r["bm25_ndcg@10"] for r in beyond)
                                          if beyond else None),
    }
    summary = {
        "study": "S3a: free reachability analysis and RM3 baseline on SciFact tune queries (no model calls)",
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_info(REPO), "python": platform.python_version(), **provenance(config),
        **rm3_provenance(rm3_config), "split": "train", "half": "tune", "dev_partition_seed": DEV_PARTITION_SEED,
        "depth": depth, "search_k": SEARCH_K, "queries": len(rows),
        "strata": {"within_depth": len(inside), "beyond_depth": len(beyond), "buckets": buckets},
        "title_reachability_beyond_depth": title, "rm3_vs_bm25_all_tune": rm3_vs_bm25,
        "rm3_by_stratum": rm3_strata,
    }
    return summary, rows


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def report(s: dict) -> str:
    st, t, r = s["strata"], s["title_reachability_beyond_depth"], s["rm3_by_stratum"]
    d, k = s["depth"], s["search_k"]
    lines = [
        "# S3a: reachability and RM3 on SciFact (tune half)", "",
        f"Model-free, deterministic, $0. Tune queries: {s['queries']}. Check half and test split not used. "
        "Sections 1-2 read labels (oracle quantities); section 3 is a label-free method scored with labels.", "",
        f"## 1. Strata by BM25's first relevant rank (first search = top {d})", "",
        f"Within top {d}: {st['within_depth']}; beyond: {st['beyond_depth']} "
        f"(ranks {d + 1}-100: {st['buckets']['beyond_to_100']}, 101-1000: {st['buckets']['beyond_to_1000']}, "
        f"beyond 1000 or none: {st['buckets']['beyond_1000']}).", "",
        f"## 2. Title reachability for the {st['beyond_depth']} beyond-{d} queries (oracle, optimistic)", "",
        "| query = a relevant document's own title | within top 10 | within top 20 |", "|---|---|---|",
        f"| that document is found | {fmt(t[f'own_within_{k}'])} | {fmt(t[f'own_within_{d}'])} |",
        f"| any relevant document is found | {fmt(t[f'any_relevant_within_{k}'])} | "
        f"{fmt(t[f'any_relevant_within_{d}'])} |", "",
        "Best over the query's relevant documents. The title is part of the indexed text (close to a known-item "
        "search), so this overstates what a model's reformulation can reach; a failure does not prove that no "
        "query works.", "",
        "## 3. RM3 (fixed defaults, not tuned) vs BM25, all tune queries", "",
        "| metric | BM25 | RM3 | RM3 - BM25 [95% CI] |", "|---|---|---|---|",
    ]
    for m, v in s["rm3_vs_bm25_all_tune"].items():
        ci = v["rm3_minus_bm25"]
        lines.append(f"| {m} | {v['bm25']:.3f} | {v['rm3']:.3f} | {ci['mean']:+.3f} [{ci['low']:+.3f}, "
                     f"{ci['high']:+.3f}] |")
    lines += [
        "", f"By stratum: of the beyond-{d} queries, RM3 puts a relevant document into its top {k} for "
        f"{fmt(r['beyond_rescued_within_search_k'])} and into its top {d} for "
        f"{fmt(r['beyond_rescued_within_depth'])}; of the within-{d} queries, RM3 loses every relevant document "
        f"from its top {d} for {fmt(r['inside_lost_from_depth'])}. Mean nDCG@10 change: within "
        f"{fmt(r['inside_ndcg@10_rm3_minus_bm25'])}, beyond {fmt(r['beyond_ndcg@10_rm3_minus_bm25'])}.",
        f"RM3 settings: {s['rm3_config']}.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--depth", type=int, default=DEPTH)
    args = parser.parse_args(argv)

    summary, rows = evaluate(args.data_dir, args.depth)
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
