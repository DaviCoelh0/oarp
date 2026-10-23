"""Score the S1 pilot from its recorded calls (free: no model calls).

    venv/bin/python experiments/retrieval_s1_pilot/analyze.py [--results DIR]

Reads results/manifest.json and results/private/calls.jsonl plus the train
labels (dev data; the test split is never loaded) and computes the contract's
section-8 metrics and its section-9 outcome rules. A query whose call failed
keeps the BM25 order the harness fell back to (scored as the harness's actual
output) and counts toward the model-error rate.

Writes results/summary.json and results/analysis.md (aggregates only) and
results/private/per_query.jsonl (label-derived, gitignored). The outcome
labels are what the pre-registered rules say; interpretation is a human step.
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
from oarp.envs.retrieval.data import DEFAULT_DATA_DIR, dev_partition, load_scifact  # noqa: E402
from oarp.envs.retrieval.metrics import first_relevant_rank, mrr_at_k, ndcg_at_k, recall_at_k  # noqa: E402
from oarp.envs.retrieval.stats import kendall_tau, paired_bootstrap_ci, spearman  # noqa: E402
from oarp.envs.retrieval.strategies import bm25_signals, rule_gate, tau_for_rate  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
R2A_SUMMARY = REPO / "experiments" / "retrieval_rerank_headroom" / "results" / "summary.json"
CONDITIONS = ("plain", "shuffled")
K = 20
BOOTSTRAP_SEED = 20261024
# Section 9 thresholds of the approved contract.
MAX_MODEL_ERROR_RATE = 0.05
MAX_FORMAT_PROBLEM_RATE = 0.10
COST_SURPRISE_PER_CALL_USD = 0.0036  # 2 x the upper estimate of $0.0018
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


def ci_label(ci: dict, positive: str, negative: str) -> str:
    return positive if ci["low"] > 0 else negative if ci["high"] < 0 else "inconclusive"


def classify(*, complete: bool, paid: bool, model_error_rate: dict[str, float],
             format_problem_rate: dict[str, float], h1: dict | None, h2: dict | None,
             mean_cost_per_call: float | None) -> dict[str, object]:
    """Section 9 of the contract, applied mechanically."""
    rules: dict[str, object] = {}
    if not paid:
        rules["run"] = "dry run: no outcome"
    elif not complete:
        rules["run"] = "incomplete (no primary or H2 result claimed)"
    elif any(rate > MAX_MODEL_ERROR_RATE for rate in model_error_rate.values()):
        rules["run"] = "invalid (model errors above 5%)"
    else:
        rules["run"] = "valid"
    rules["format_problem"] = any(rate > MAX_FORMAT_PROBLEM_RATE for rate in format_problem_rate.values())
    claimable = rules["run"] == "valid"
    rules["H1"] = ci_label(h1, "positive", "negative") if claimable and h1 else None
    rules["H2"] = ci_label(h2, "shuffled better", "plain better") if claimable and h2 else None
    rules["cost_surprise"] = mean_cost_per_call is not None and mean_cost_per_call > COST_SURPRISE_PER_CALL_USD
    return rules


def _p90(values: list[float]) -> float | None:
    return sorted(values)[int(0.9 * (len(values) - 1))] if values else None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def presentation_rho(row: dict) -> float | None:
    """Spearman's rho between where each candidate was shown and where the model put it."""
    if row["status"] not in ("valid", "repaired") or len(row["bm25_top"]) < 2:
        return None
    shown_docs = [row["bm25_top"][p] for p in row["shown_bm25_positions"]]
    model_rank = {doc: r for r, doc in enumerate(row["ranking_top"])}
    return spearman(list(range(len(shown_docs))), [model_rank[d] for d in shown_docs])


def analyze(results: Path, data_dir: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((results / "manifest.json").read_text())
    rows = [json.loads(line) for line in (results / "private" / "calls.jsonl").read_text().splitlines() if line]
    sample = json.loads((results / "private" / "sample.json").read_text())
    split = load_scifact(data_dir, "train")
    qrels = split.qrels

    pass1 = {c: {r["query_id"]: r for r in rows if r["pass"] == 1 and r["condition"] == c} for c in CONDITIONS}
    queries = [q for q in sample["sample"] if q in pass1["plain"] or q in pass1["shuffled"]]
    bm25_top = {q: (pass1["plain"].get(q) or pass1["shuffled"][q])["bm25_top"] for q in queries}
    s0 = {q: metrics(bm25_top[q], qrels[q]) for q in queries}
    oracle = {q: metrics(oracle_rerank(bm25_top[q], qrels[q], K), qrels[q]) for q in queries}
    s1 = {c: {q: metrics(pass1[c][q]["ranking_top"], qrels[q]) for q in queries if q in pass1[c]} for c in CONDITIONS}
    r1_relevant = {q: rank1_relevant(bm25_top[q], qrels[q]) for q in queries}

    def diff_ci(a: dict, b: dict) -> dict | None:
        common = [q for q in queries if q in a and q in b]
        return paired_bootstrap_ci([a[q]["ndcg@10"] - b[q]["ndcg@10"] for q in common],
                                   seed=BOOTSTRAP_SEED) if common else None

    h1 = diff_ci(s1["plain"], s0)
    h2 = diff_ci(s1["shuffled"], s1["plain"])
    shuffled_vs_s0 = diff_ci(s1["shuffled"], s0)

    # Headroom-bucket mix: the sample vs the whole tune half (BM25 is deterministic).
    index = BM25Index(split.corpus, BM25Config())
    tune = dev_partition(sorted(split.queries))["tune"]
    full = {q: [d for d, _ in index.rank(split.queries[q])] for q in tune}
    scored = {q: index.rank(split.queries[q], 2) for q in queries}

    def mix(ids: list[str]) -> dict[str, float]:
        counts = Counter(bucket(first_relevant_rank(full[q], qrels[q])) for q in ids)
        return {b: counts.get(b, 0) / len(ids) for b in BUCKETS} if ids else {}

    per_condition = {}
    for c in CONDITIONS:
        ids = [q for q in queries if q in s1[c]]
        calls = [r for r in rows if r["condition"] == c]
        p1 = [pass1[c][q] for q in ids]
        statuses = Counter(r["status"] for r in calls)
        mean = {m: _mean([s1[c][q][m] for q in ids]) for m in ("ndcg@10", "recall@10", "mrr@10")}
        s0_mean, oracle_mean = _mean([s0[q]["ndcg@10"] for q in ids]), _mean([oracle[q]["ndcg@10"] for q in ids])
        rel1 = [q for q in ids if r1_relevant[q]]
        usage = [r["usage"] for r in calls if r["usage"]]
        costs = [r["estimated_usd"] for r in calls if r["estimated_usd"] is not None]
        rhos = [rho for rho in (presentation_rho(r) for r in p1) if rho is not None]
        per_condition[c] = {
            "queries": len(ids), "calls": len(calls), **{f"{m}": v for m, v in mean.items()},
            "s0_ndcg@10": s0_mean, "oracle20_ndcg@10": oracle_mean,
            "share_of_ceiling": (mean["ndcg@10"] - s0_mean) / (oracle_mean - s0_mean)
            if ids and oracle_mean != s0_mean else None,
            "harm_rate": _mean([s1[c][q]["ndcg@10"] < s0[q]["ndcg@10"] for q in ids]),
            "harm_rate_rank1_relevant": _mean([s1[c][q]["ndcg@10"] < s0[q]["ndcg@10"] for q in rel1]),
            "by_bucket": {b: {"queries": len(bq), "s0": _mean([s0[q]["ndcg@10"] for q in bq]),
                              "s1": _mean([s1[c][q]["ndcg@10"] for q in bq])}
                          for b in BUCKETS
                          for bq in [[q for q in ids if bucket(first_relevant_rank(full[q], qrels[q])) == b]]},
            "copied_presentation_order_rate": _mean([r["copied_presentation_order"] for r in p1]),
            "matches_bm25_order_rate": _mean([r["matches_bm25_order"] for r in p1]),
            "mean_spearman_shown_vs_model_rank": _mean(rhos),
            "status_counts": dict(statuses),
            "model_error_rate": statuses.get("model_error", 0) / len(calls) if calls else 0.0,
            "format_problem_rate": (sum(r["status"] in ("failed", "repaired") for r in p1) / len(p1)) if p1 else 0.0,
            "repairs": {k: sum(r[k] for r in calls) for k in ("duplicates", "out_of_range", "missing")},
            "truncations": sum(r["error_kind"] == "ModelOutputTruncated" for r in calls),
            "error_kinds": dict(Counter(r["error_kind"] for r in calls if r["error_kind"])),
            "input_tokens": {"mean": _mean([u["input_tokens"] for u in usage]),
                             "p90": _p90([u["input_tokens"] for u in usage])},
            "output_tokens": {"mean": _mean([u["output_tokens"] for u in usage]),
                              "p90": _p90([u["output_tokens"] for u in usage])},
            "estimated_usd": {"total": sum(costs), "per_call_mean": _mean(costs)},
            "latency_s": {"median": statistics.median([r["latency_s"] for r in calls]) if calls else None,
                          "p90": _p90([r["latency_s"] for r in calls])},
            "stability": stability([r for r in calls if r["query_id"] in set(sample["stability"])], qrels),
            "exploratory": exploratory({q: s0[q]["ndcg@10"] for q in ids}, {q: s1[c][q]["ndcg@10"] for q in ids},
                                       {q: not r1_relevant[q] for q in ids},
                                       {q: bm25_signals(scored[q])["relative_gap"] for q in ids}),
        }

    all_costs = [r["estimated_usd"] for r in rows if r["estimated_usd"] is not None]
    complete = manifest["completed_calls"] == manifest["scheduled_calls"] and not manifest["stopped_by_budget"]
    outcome = classify(complete=complete, paid=manifest["mode"] == "paid",
                       model_error_rate={c: v["model_error_rate"] for c, v in per_condition.items()},
                       format_problem_rate={c: v["format_problem_rate"] for c, v in per_condition.items()},
                       h1=h1, h2=h2, mean_cost_per_call=_mean(all_costs))
    r2a = json.loads(R2A_SUMMARY.read_text())["estimated_cost"]["rerank"] if R2A_SUMMARY.is_file() else None
    summary = {
        "study": manifest["study"], "mode": manifest["mode"], "git_at_run": manifest["git"],
        "model": manifest["model"], "calls": {"scheduled": manifest["scheduled_calls"],
                                              "completed": manifest["completed_calls"]},
        "stopped_by_budget": manifest["stopped_by_budget"], "estimated_spend_usd": manifest["estimated_spend_usd"],
        "queries": len(queries), "bucket_mix": {"sample": mix(queries), "tune_half": mix(tune)},
        "H1_plain_minus_s0": h1, "H2_shuffled_minus_plain": h2, "shuffled_minus_s0": shuffled_vs_s0,
        "per_condition": per_condition, "r2a_cost_estimate": r2a, "outcome_rules": outcome,
    }
    per_query = [{"query_id": q, "bm25_bucket": bucket(first_relevant_rank(full[q], qrels[q])),
                  "s0_ndcg@10": s0[q]["ndcg@10"], "oracle20_ndcg@10": oracle[q]["ndcg@10"],
                  **{f"{c}_ndcg@10": s1[c][q]["ndcg@10"] for c in CONDITIONS if q in s1[c]}} for q in queries]
    return summary, per_query


def stability(calls: list[dict], qrels: dict) -> dict:
    by_query: dict[str, list[dict]] = {}
    for r in calls:
        by_query.setdefault(r["query_id"], []).append(r)
    full = {q: sorted(rs, key=lambda r: r["pass"]) for q, rs in by_query.items() if len(rs) >= 3}
    if not full:
        return {"queries": 0}
    sds = [statistics.stdev(ndcg_at_k(r["ranking_top"], qrels[q], 10) for r in rs) for q, rs in full.items()]
    top1 = [len({r["ranking_top"][0] for r in rs}) == 1 for rs in full.values()]
    taus = [kendall_tau(a["ranking_top"], b["ranking_top"]) for rs in full.values() for a, b in combinations(rs, 2)
            if len(a["ranking_top"]) > 1]
    return {"queries": len(full), "mean_sd_ndcg@10": statistics.fmean(sds), "top1_agreement": statistics.fmean(top1),
            "mean_kendall_tau_top20": _mean(taus)}


def exploratory(s0: dict[str, float], s1: dict[str, float], oracle_flags: dict[str, bool],
                gaps: dict[str, float]) -> dict:
    """Oracle gate x S1 and S2 x S1 at the oracle gate's rate; on the same queries (exploratory)."""
    if not s0:
        return {}
    rate = statistics.fmean(oracle_flags.values())
    tau = tau_for_rate(list(gaps.values()), rate)
    flags = rule_gate(gaps, tau)
    mixed = lambda f: statistics.fmean(s1[q] if f[q] else s0[q] for q in s0)  # noqa: E731
    return {"oracle_gate_rate": rate, "oracle_gate_x_s1_ndcg@10": mixed(oracle_flags),
            "s2_tau": tau, "s2_rate": statistics.fmean(flags.values()), "s2_x_s1_ndcg@10": mixed(flags)}


def _f(value, spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value, spec)


def report(s: dict) -> str:
    def ci(c: dict | None) -> str:
        return "n/a" if c is None else f"{c['mean']:+.3f} [{c['low']:+.3f}, {c['high']:+.3f}] (n = {c['n']})"

    lines = [f"# S1 pilot ({s['mode']})", "",
             f"Calls: {s['calls']['completed']}/{s['calls']['scheduled']}; stopped by budget: {s['stopped_by_budget']}; "
             f"estimated spend ${s['estimated_spend_usd']:.4f}. Queries: {s['queries']}. Outcome rules: {s['outcome_rules']}.",
             "", "## Paired nDCG@10 differences (pass 1, bootstrap 95% CI)", "",
             f"- H1 S1-plain − S0: {ci(s['H1_plain_minus_s0'])}",
             f"- H2 S1-shuffled − S1-plain: {ci(s['H2_shuffled_minus_plain'])}",
             f"- S1-shuffled − S0: {ci(s['shuffled_minus_s0'])}", "",
             "## Per condition", "",
             "| | plain | shuffled |", "|---|---|---|"]
    pc = s["per_condition"]
    for label, get in [
        ("nDCG@10", lambda v: _f(v["ndcg@10"])), ("Recall@10", lambda v: _f(v["recall@10"])),
        ("MRR@10", lambda v: _f(v["mrr@10"])), ("S0 nDCG@10", lambda v: _f(v["s0_ndcg@10"])),
        ("perfect top-20 nDCG@10", lambda v: _f(v["oracle20_ndcg@10"])),
        ("share of ceiling", lambda v: _f(v["share_of_ceiling"])), ("harm rate", lambda v: _f(v["harm_rate"])),
        ("harm rate (rank 1 relevant)", lambda v: _f(v["harm_rate_rank1_relevant"])),
        ("copied presentation order", lambda v: _f(v["copied_presentation_order_rate"])),
        ("final top 20 = BM25", lambda v: _f(v["matches_bm25_order_rate"])),
        ("Spearman shown vs model rank", lambda v: _f(v["mean_spearman_shown_vs_model_rank"])),
        ("model-error rate", lambda v: _f(v["model_error_rate"])),
        ("format-problem rate", lambda v: _f(v["format_problem_rate"])), ("truncations", lambda v: str(v["truncations"])),
        ("status counts", lambda v: str(v["status_counts"])),
        ("input tokens mean / p90", lambda v: f"{_f(v['input_tokens']['mean'], '.0f')} / {_f(v['input_tokens']['p90'], '.0f')}"),
        ("output tokens mean / p90", lambda v: f"{_f(v['output_tokens']['mean'], '.0f')} / {_f(v['output_tokens']['p90'], '.0f')}"),
        ("est. $ total / per call", lambda v: f"{v['estimated_usd']['total']:.4f} / {_f(v['estimated_usd']['per_call_mean'], '.5f')}"),
        ("latency s median / p90", lambda v: f"{_f(v['latency_s']['median'], '.1f')} / {_f(v['latency_s']['p90'], '.1f')}"),
        ("stability", lambda v: str(v["stability"])), ("exploratory gates", lambda v: str(v["exploratory"])),
    ]:
        lines.append(f"| {label} | {get(pc['plain'])} | {get(pc['shuffled'])} |")
    lines += ["", f"Bucket mix (sample vs tune half): {s['bucket_mix']}",
              f"R2a cost estimate for comparison: {s['r2a_cost_estimate']}"]
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
