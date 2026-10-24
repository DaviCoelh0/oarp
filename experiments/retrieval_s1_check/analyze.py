"""Score the S1 check-half study from its recorded calls (free: no model calls).

    venv/bin/python experiments/retrieval_s1_check/analyze.py [--results DIR]

Computes the approved contract's metrics and applies its outcome rules
mechanically:
- primary H1c: S1-plain - S0 on all check queries (pass 1), paired bootstrap
  95% CI; a declined or failed query keeps the BM25 order (scored as the
  harness's actual output);
- pre-specified sensitivity: the same, excluding safety-declined queries;
- declines (ModelRefused) counted separately from other model errors;
- descriptive metrics, stability (next to the pilot's), and, for later gate
  studies, S2 x S1 across rerank rates and the oracle gate x S1 (no tau chosen).

Writes results/summary.json and results/analysis.md (aggregates only) and
results/private/per_query.jsonl (label-derived, gitignored).
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from oarp.envs.retrieval.analysis import oracle_rerank, rank1_relevant  # noqa: E402
from oarp.envs.retrieval.bm25 import BM25Config, BM25Index  # noqa: E402
from oarp.envs.retrieval.data import DEFAULT_DATA_DIR, load_scifact  # noqa: E402
from oarp.envs.retrieval.metrics import first_relevant_rank, mrr_at_k, ndcg_at_k, recall_at_k  # noqa: E402
from oarp.envs.retrieval.stats import kendall_tau, paired_bootstrap_ci  # noqa: E402
from oarp.envs.retrieval.strategies import bm25_signals, rule_gate, tau_for_rate  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PILOT_SUMMARY = REPO / "experiments" / "retrieval_s1_pilot" / "results" / "summary.json"
K = 20
BOOTSTRAP_SEED = 20261024
DECLINE = "ModelRefused"
# Section 8 of the approved contract.
MAX_OTHER_ERROR_RATE = 0.05
DECLINE_WARNING_RATE = 0.10
COST_SURPRISE_PER_CALL_USD = 0.0026  # 2 x the pilot's measured mean of $0.00132
RATES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.8)
BUCKETS = ("rank 1", "ranks 2-10", "ranks 11-20", "beyond 20")


def bucket(rank: int | None) -> str:
    if rank == 1:
        return BUCKETS[0]
    if rank is not None and rank <= 10:
        return BUCKETS[1]
    if rank is not None and rank <= 20:
        return BUCKETS[2]
    return BUCKETS[3]


def metrics(ranking: list[str], judged: dict[str, int]) -> dict[str, float]:
    return {"ndcg@10": ndcg_at_k(ranking, judged, 10), "recall@10": recall_at_k(ranking, judged, 10),
            "mrr@10": mrr_at_k(ranking, judged, 10)}


def classify(*, complete: bool, paid: bool, other_error_rate: float, decline_rate: float, h1: dict | None,
             mean_cost_per_call: float | None) -> dict[str, object]:
    """Section 8 of the contract, applied mechanically."""
    if not paid:
        run = "dry run: no outcome"
    elif not complete:
        run = "incomplete (no primary result claimed)"
    elif other_error_rate > MAX_OTHER_ERROR_RATE:
        run = "invalid (non-decline model errors above 5%)"
    else:
        run = "valid"
    h1c = None
    if run == "valid" and h1:
        h1c = "confirmed" if h1["low"] > 0 else "reversed" if h1["high"] < 0 else "not confirmed"
    return {"run": run, "H1c": h1c, "decline_warning": decline_rate > DECLINE_WARNING_RATE,
            "cost_surprise": mean_cost_per_call is not None and mean_cost_per_call > COST_SURPRISE_PER_CALL_USD}


def _mean(values) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def _p90(values: list[float]) -> float | None:
    return sorted(values)[int(0.9 * (len(values) - 1))] if values else None


def stability(calls: list[dict], qrels: dict) -> dict:
    by_query: dict[str, list[dict]] = {}
    for r in calls:
        by_query.setdefault(r["query_id"], []).append(r)
    full = {q: sorted(rs, key=lambda r: r["pass"]) for q, rs in by_query.items() if len(rs) >= 3}
    if not full:
        return {"queries": 0}
    sds = [statistics.stdev(ndcg_at_k(r["ranking_top"], qrels[q], 10) for r in rs) for q, rs in full.items()]
    taus = [kendall_tau(a["ranking_top"], b["ranking_top"]) for rs in full.values()
            for a, b in combinations(rs, 2) if len(a["ranking_top"]) > 1]
    return {"queries": len(full), "mean_sd_ndcg@10": statistics.fmean(sds),
            "queries_with_varying_ndcg": sum(sd > 0 for sd in sds),
            "top1_agreement": statistics.fmean(len({r["ranking_top"][0] for r in rs}) == 1 for rs in full.values()),
            "mean_kendall_tau_top20": _mean(taus)}


def analyze(results: Path, data_dir: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((results / "manifest.json").read_text())
    rows = [json.loads(line) for line in (results / "private" / "calls.jsonl").read_text().splitlines() if line]
    sample = json.loads((results / "private" / "sample.json").read_text())
    split = load_scifact(data_dir, "train")
    qrels = split.qrels

    pass1 = {r["query_id"]: r for r in rows if r["pass"] == 1}
    queries = [q for q in sample["queries"] if q in pass1]
    s0 = {q: metrics(pass1[q]["bm25_top"], qrels[q]) for q in queries}
    s1 = {q: metrics(pass1[q]["ranking_top"], qrels[q]) for q in queries}
    oracle = {q: metrics(oracle_rerank(pass1[q]["bm25_top"], qrels[q], K), qrels[q]) for q in queries}
    declined = {q for q in queries if pass1[q]["error_kind"] == DECLINE}

    def diff_ci(ids: list[str]) -> dict | None:
        return paired_bootstrap_ci([s1[q]["ndcg@10"] - s0[q]["ndcg@10"] for q in ids],
                                   seed=BOOTSTRAP_SEED) if ids else None

    h1 = diff_ci(queries)
    h1_no_declines = diff_ci([q for q in queries if q not in declined])

    index = BM25Index(split.corpus, BM25Config())
    full = {q: [d for d, _ in index.rank(split.queries[q])] for q in queries}
    gaps = {q: bm25_signals(index.rank(split.queries[q], 2))["relative_gap"] for q in queries}
    first = {q: first_relevant_rank(full[q], qrels[q]) for q in queries}

    errors = Counter(r["error_kind"] for r in rows if r["status"] == "model_error")
    declines = errors.get(DECLINE, 0)
    other_errors = sum(errors.values()) - declines
    usage = [r["usage"] for r in rows if r["usage"]]
    costs = [r["estimated_usd"] for r in rows if r["estimated_usd"] is not None]
    s0_mean, s1_mean, oracle_mean = (_mean(m[q]["ndcg@10"] for q in queries) for m in (s0, s1, oracle))
    rel1 = [q for q in queries if rank1_relevant(pass1[q]["bm25_top"], qrels[q])]
    s0_n, s1_n = ({q: m[q]["ndcg@10"] for q in queries} for m in (s0, s1))
    oracle_flags = {q: q not in set(rel1) for q in queries}

    def mixed(flags: dict[str, bool]) -> float:
        return statistics.fmean(s1_n[q] if flags[q] else s0_n[q] for q in queries)

    s2_curve = []
    for rate in RATES:
        tau = tau_for_rate(list(gaps.values()), rate) if queries else None
        flags = rule_gate(gaps, tau) if queries else {}
        s2_curve.append({"target_rate": rate, "tau": tau, "rate": _mean(flags.values()),
                         "ndcg@10": mixed(flags) if queries else None})
    pilot = json.loads(PILOT_SUMMARY.read_text()) if PILOT_SUMMARY.is_file() else None
    summary = {
        "study": manifest["study"], "mode": manifest["mode"], "git_at_run": manifest["git"],
        "pilot_commit": manifest.get("pilot_commit"), "model": manifest["model"],
        "calls": {"scheduled": manifest["scheduled_calls"], "completed": manifest["completed_calls"]},
        "stopped_by_budget": manifest["stopped_by_budget"], "estimated_spend_usd": manifest["estimated_spend_usd"],
        "queries": len(queries),
        "H1c_s1_minus_s0": h1, "H1c_excluding_declined": h1_no_declines,
        "declined_queries": len(declined),
        "means": {"s0_ndcg@10": s0_mean, "s1_ndcg@10": s1_mean, "oracle20_ndcg@10": oracle_mean,
                  "s1_recall@10": _mean(s1[q]["recall@10"] for q in queries),
                  "s0_recall@10": _mean(s0[q]["recall@10"] for q in queries),
                  "s1_mrr@10": _mean(s1[q]["mrr@10"] for q in queries),
                  "s0_mrr@10": _mean(s0[q]["mrr@10"] for q in queries)},
        "share_of_ceiling": (s1_mean - s0_mean) / (oracle_mean - s0_mean)
        if queries and oracle_mean != s0_mean else None,
        "improved_harmed_unchanged": [sum(s1_n[q] > s0_n[q] for q in queries), sum(s1_n[q] < s0_n[q] for q in queries),
                                      sum(s1_n[q] == s0_n[q] for q in queries)],
        "harm_rate": _mean(s1_n[q] < s0_n[q] for q in queries),
        "harm_rate_rank1_relevant": _mean(s1_n[q] < s0_n[q] for q in rel1),
        "by_bucket": {b: {"queries": len(ids), "s0": _mean(s0_n[q] for q in ids), "s1": _mean(s1_n[q] for q in ids)}
                      for b in BUCKETS for ids in [[q for q in queries if bucket(first[q]) == b]]},
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "declines": declines, "decline_rate": declines / len(rows) if rows else 0.0,
        "decline_categories": dict(Counter((r["error"] or "").split("category: ")[-1].rstrip(")")
                                           for r in rows if r["error_kind"] == DECLINE)),
        "other_model_errors": other_errors, "other_error_rate": other_errors / len(rows) if rows else 0.0,
        "error_kinds": dict(errors),
        "repairs": {k: sum(r[k] for r in rows) for k in ("duplicates", "out_of_range", "missing")},
        "truncations": errors.get("ModelOutputTruncated", 0),
        "input_tokens": {"mean": _mean(u["input_tokens"] for u in usage), "p90": _p90([u["input_tokens"] for u in usage])},
        "output_tokens": {"mean": _mean(u["output_tokens"] for u in usage), "p90": _p90([u["output_tokens"] for u in usage])},
        "estimated_usd": {"total": sum(costs), "per_call_mean": _mean(costs)},
        "latency_s": {"median": statistics.median(r["latency_s"] for r in rows) if rows else None,
                      "p90": _p90([r["latency_s"] for r in rows])},
        "stability": stability([r for r in rows if r["query_id"] in set(sample["stability"])], qrels),
        "pilot_stability": pilot["per_condition"]["plain"]["stability"] if pilot else None,
        "for_gate_studies": {"oracle_gate_rate": _mean(oracle_flags.values()),
                             "oracle_gate_x_s1_ndcg@10": mixed(oracle_flags) if queries else None,
                             "s2_x_s1_curve": s2_curve},
    }
    summary["outcome_rules"] = classify(
        complete=manifest["completed_calls"] == manifest["scheduled_calls"] and not manifest["stopped_by_budget"],
        paid=manifest["mode"] == "paid", other_error_rate=summary["other_error_rate"],
        decline_rate=summary["decline_rate"], h1=h1, mean_cost_per_call=summary["estimated_usd"]["per_call_mean"])
    per_query = [{"query_id": q, "bm25_bucket": bucket(first[q]), "s0_ndcg@10": s0_n[q], "s1_ndcg@10": s1_n[q],
                  "oracle20_ndcg@10": oracle[q]["ndcg@10"], "declined": q in declined} for q in queries]
    return summary, per_query


def _f(value, spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value, spec)


def report(s: dict) -> str:
    def ci(c: dict | None) -> str:
        return "n/a" if c is None else f"{c['mean']:+.3f} [{c['low']:+.3f}, {c['high']:+.3f}] (n = {c['n']})"

    m = s["means"]
    lines = [
        f"# S1 check-half confirmation ({s['mode']})", "",
        f"Calls: {s['calls']['completed']}/{s['calls']['scheduled']}; stopped by budget: {s['stopped_by_budget']}; "
        f"estimated spend ${s['estimated_spend_usd']:.4f}. Queries: {s['queries']}. Outcome rules: {s['outcome_rules']}.", "",
        f"- H1c S1-plain − S0: {ci(s['H1c_s1_minus_s0'])}",
        f"- Excluding {s['declined_queries']} declined queries (pre-specified): {ci(s['H1c_excluding_declined'])}", "",
        "| | S0 | S1-plain | perfect top-20 |", "|---|---|---|---|",
        f"| nDCG@10 | {_f(m['s0_ndcg@10'])} | {_f(m['s1_ndcg@10'])} | {_f(m['oracle20_ndcg@10'])} |",
        f"| Recall@10 | {_f(m['s0_recall@10'])} | {_f(m['s1_recall@10'])} | |",
        f"| MRR@10 | {_f(m['s0_mrr@10'])} | {_f(m['s1_mrr@10'])} | |", "",
        f"Share of ceiling {_f(s['share_of_ceiling'])}; improved / harmed / unchanged {s['improved_harmed_unchanged']}; "
        f"harm rate {_f(s['harm_rate'])} (rank 1 relevant: {_f(s['harm_rate_rank1_relevant'])}).",
        f"By bucket: {s['by_bucket']}.",
        f"Status {s['status_counts']}; declines {s['declines']} ({_f(s['decline_rate'])}, {s['decline_categories']}); "
        f"other model errors {s['other_model_errors']} ({_f(s['other_error_rate'])}); repairs {s['repairs']}; "
        f"truncations {s['truncations']}.",
        f"Tokens in {s['input_tokens']}, out {s['output_tokens']}; est. $ {s['estimated_usd']}; latency {s['latency_s']}.",
        f"Stability: {s['stability']} (pilot: {s['pilot_stability']}).",
        f"For gate studies: {s['for_gate_studies']}.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)
    summary, per_query = analyze(args.results, args.data_dir)
    (args.results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.results / "analysis.md").write_text(report(summary))
    with (args.results / "private" / "per_query.jsonl").open("w") as f:
        for row in per_query:
            f.write(json.dumps(row) + "\n")
    print(report(summary))
    return summary


if __name__ == "__main__":
    main()
