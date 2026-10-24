"""Score the S2b gate pilot (free: no model calls).

    venv/bin/python experiments/retrieval_s2b_pilot/analyze.py [--results DIR] [--pilot-results DIR]

Reads the gate records and, read-only, the S1 pilot's pass-1 S1-plain records
(rankings, rerank cost and latency per query), plus the train labels (dev
data). Applies the approved contract mechanically:

- correctness: a decision is right iff (rerank) == (BM25 rank 1 not relevant);
- S2 at a matched rate: tau_for_rate on these queries' relative gaps, set to
  each gate condition's pass-1 rerank rate (achieved rate reported);
- primary: pass 1, S2b-medium correctness - matched S2 correctness, paired
  bootstrap 95% CI;
- gate quality, quality and cost of every strategy (gated rankings from
  strategies.combine over the S1 pilot's rankings), flip rate, effort
  comparison, and the section-9 outcome rules.

A gate call that fails or is declined counts as "rerank" (the committed
fallback). Where the S1 pilot has no rerank cost for a query (its call was
declined), the pilot's mean rerank cost is used (stated in the summary).

Writes results/summary.json and results/analysis.md (aggregates only) and
results/private/per_query.jsonl (label-derived, gitignored).
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from oarp.envs.retrieval.analysis import gate_precision_recall, rank1_relevant  # noqa: E402
from oarp.envs.retrieval.bm25 import BM25Config, BM25Index  # noqa: E402
from oarp.envs.retrieval.data import DEFAULT_DATA_DIR, load_scifact  # noqa: E402
from oarp.envs.retrieval.gate import flip_rate  # noqa: E402
from oarp.envs.retrieval.metrics import ndcg_at_k  # noqa: E402
from oarp.envs.retrieval.stats import paired_bootstrap_ci  # noqa: E402
from oarp.envs.retrieval.strategies import bm25_signals, combine, rule_gate, tau_for_rate  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PILOT_RESULTS = REPO / "experiments" / "retrieval_s1_pilot" / "results"
EFFORTS = ("medium", "low")
BOOTSTRAP_SEED = 20261024
DECLINE = "ModelRefused"
# Section 9 of the approved contract.
MAX_OTHER_ERROR_RATE = 0.05
MAX_UNPARSEABLE_RATE = 0.10
MAX_FLIP_RATE = 0.20


def _mean(values) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def ci_label(ci: dict | None, better: str, not_better: str, worse: str) -> str | None:
    if ci is None:
        return None
    return better if ci["low"] > 0 else worse if ci["high"] < 0 else not_better


def classify(*, complete: bool, paid: bool, other_error_rate: dict[str, float], unparseable_rate: dict[str, float],
             primary: dict | None, gate_cost_per_query: float | None, s1_cost_per_query: float | None,
             flip_medium: float | None) -> dict[str, object]:
    """Section 9 of the contract, applied mechanically."""
    if not paid:
        run = "dry run: no outcome"
    elif not complete:
        run = "incomplete (no primary result claimed)"
    elif any(rate > MAX_OTHER_ERROR_RATE for rate in other_error_rate.values()):
        run = "invalid (non-decline model errors above 5%)"
    else:
        run = "valid"
    claimable = run == "valid"
    return {
        "run": run,
        "format_problem": any(rate > MAX_UNPARSEABLE_RATE for rate in unparseable_rate.values()),
        "primary": ci_label(primary, "gate better", "gate not better", "gate worse") if claimable else None,
        "no_saving": None if gate_cost_per_query is None or s1_cost_per_query is None
        else gate_cost_per_query >= s1_cost_per_query,
        "unstable": flip_medium is not None and flip_medium > MAX_FLIP_RATE,
    }


def analyze(results: Path, pilot_results: Path, data_dir: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((results / "manifest.json").read_text())
    rows = [json.loads(line) for line in (results / "private" / "calls.jsonl").read_text().splitlines() if line]
    pilot_rows = [json.loads(line) for line in (pilot_results / "private" / "calls.jsonl").read_text().splitlines()
                  if line]
    s1_rec = {r["query_id"]: r for r in pilot_rows if r["pass"] == 1 and r["condition"] == "plain"}
    split = load_scifact(data_dir, "train")
    qrels = split.qrels

    pass1 = {e: {r["query_id"]: r for r in rows if r["pass"] == 1 and r["effort"] == e} for e in EFFORTS}
    queries = [q for q in s1_rec if all(q in pass1[e] for e in EFFORTS)]
    bm25 = {q: s1_rec[q]["bm25_top"] for q in queries}
    s1 = {q: s1_rec[q]["ranking_top"] for q in queries}
    ideal = {q: not rank1_relevant(bm25[q], qrels[q]) for q in queries}  # True = should rerank
    index = BM25Index(split.corpus, BM25Config())
    gaps = {q: bm25_signals(index.rank(split.queries[q], 2))["relative_gap"] for q in queries}

    known = [r["estimated_usd"] for r in s1_rec.values() if r["estimated_usd"] is not None]
    mean_rerank_cost = _mean(known) or 0.0
    rerank_cost = {q: s1_rec[q]["estimated_usd"] if s1_rec[q]["estimated_usd"] is not None else mean_rerank_cost
                   for q in queries}
    rerank_latency = {q: s1_rec[q]["latency_s"] for q in queries}
    ndcg = lambda ranking, q: ndcg_at_k(ranking, qrels[q], 10)  # noqa: E731
    s0_n = {q: ndcg(bm25[q], q) for q in queries}
    s1_n = {q: ndcg(s1[q], q) for q in queries}

    def strategy(flags: dict[str, bool], gate_cost: dict[str, float], gate_latency: dict[str, float]) -> dict:
        final = combine(bm25, s1, flags)
        n = {q: ndcg(final[q], q) for q in queries}
        diff = paired_bootstrap_ci([n[q] - s1_n[q] for q in queries], seed=BOOTSTRAP_SEED) if queries else None
        s0_mean, s1_mean = _mean(s0_n.values()), _mean(s1_n.values())
        mean = _mean(n.values())
        return {"rerank_rate": _mean(flags.values()), "ndcg@10": mean, "minus_s1": diff,
                "share_of_s1_gain": (mean - s0_mean) / (s1_mean - s0_mean) if queries and s1_mean != s0_mean else None,
                "cost_per_query_usd": _mean(gate_cost[q] + (rerank_cost[q] if flags[q] else 0.0) for q in queries),
                "latency_per_query_s": _mean(gate_latency[q] + (rerank_latency[q] if flags[q] else 0.0)
                                             for q in queries)}

    zero = {q: 0.0 for q in queries}
    strategies = {"S0": strategy({q: False for q in queries}, zero, zero),
                  "S1": strategy({q: True for q in queries}, zero, zero),
                  "oracle_gate": strategy(ideal, zero, zero)}
    gates, primary_by_effort = {}, {}
    for e in EFFORTS:
        flags = {q: pass1[e][q]["rerank"] for q in queries}
        cost = {q: pass1[e][q]["estimated_usd"] or 0.0 for q in queries}
        latency = {q: pass1[e][q]["latency_s"] for q in queries}
        rate = _mean(flags.values())
        tau = tau_for_rate(list(gaps.values()), rate) if queries else None
        s2_flags = rule_gate(gaps, tau) if queries else {}
        correct = {q: flags[q] == ideal[q] for q in queries}
        s2_correct = {q: s2_flags[q] == ideal[q] for q in queries}
        primary_by_effort[e] = paired_bootstrap_ci([correct[q] - s2_correct[q] for q in queries],
                                                   seed=BOOTSTRAP_SEED) if queries else None
        calls = [r for r in rows if r["effort"] == e]
        errors = Counter(r["error_kind"] for r in calls if r["status"] == "model_error")
        repeated = {}
        for r in calls:
            repeated.setdefault(r["query_id"], []).append(r["rerank"])
        repeated = {q: d for q, d in repeated.items() if len(d) >= 2}
        usage = [r["usage"] for r in calls if r["usage"]]
        strategies[f"S2b_{e}"] = strategy(flags, cost, latency)
        strategies[f"S2_matched_{e}"] = strategy(s2_flags, zero, zero)
        gates[e] = {
            "rerank_rate": rate, "s2_tau": tau, "s2_rate": _mean(s2_flags.values()),
            "correct_rate": _mean(correct.values()), "s2_correct_rate": _mean(s2_correct.values()),
            "gate_minus_s2_correct": primary_by_effort[e],
            **gate_precision_recall({q: not flags[q] for q in queries}, qrels, bm25),
            "status_counts": dict(Counter(r["status"] for r in calls)),
            "unparseable_rate_pass1": _mean(pass1[e][q]["status"] == "unparseable" for q in queries) or 0.0,
            "declines": errors.get(DECLINE, 0),
            "other_error_rate": (sum(errors.values()) - errors.get(DECLINE, 0)) / len(calls) if calls else 0.0,
            "flip_rate": flip_rate(repeated) if repeated else None,
            "input_tokens_mean": _mean(u["input_tokens"] for u in usage),
            "output_tokens_mean": _mean(u["output_tokens"] for u in usage),
            "gate_cost_per_call_usd": _mean(r["estimated_usd"] for r in calls if r["estimated_usd"] is not None),
            "gate_latency_median_s": statistics.median(r["latency_s"] for r in calls) if calls else None,
        }
    low_minus_medium = paired_bootstrap_ci(
        [(pass1["low"][q]["rerank"] == ideal[q]) - (pass1["medium"][q]["rerank"] == ideal[q]) for q in queries],
        seed=BOOTSTRAP_SEED) if queries else None
    summary = {
        "study": manifest["study"], "mode": manifest["mode"], "git_at_run": manifest["git"],
        "s1_pilot": manifest["s1_pilot"], "model": manifest["model"],
        "calls": {"scheduled": manifest["scheduled_calls"], "completed": manifest["completed_calls"]},
        "stopped_by_budget": manifest["stopped_by_budget"], "estimated_spend_usd": manifest["estimated_spend_usd"],
        "queries": len(queries), "oracle_rerank_rate": _mean(ideal.values()),
        "primary_medium_minus_s2_correct": primary_by_effort.get("medium"),
        "gates": gates, "low_minus_medium_correct": low_minus_medium, "strategies": strategies,
        "rerank_cost_note": f"S1 pilot rerank cost per query; declined pilot calls use the pilot mean "
                            f"${mean_rerank_cost:.5f}",
    }
    summary["outcome_rules"] = classify(
        complete=manifest["completed_calls"] == manifest["scheduled_calls"] and not manifest["stopped_by_budget"],
        paid=manifest["mode"] == "paid",
        other_error_rate={e: g["other_error_rate"] for e, g in gates.items()},
        unparseable_rate={e: g["unparseable_rate_pass1"] for e, g in gates.items()},
        primary=summary["primary_medium_minus_s2_correct"],
        gate_cost_per_query=strategies["S2b_medium"]["cost_per_query_usd"],
        s1_cost_per_query=strategies["S1"]["cost_per_query_usd"], flip_medium=gates["medium"]["flip_rate"])
    per_query = [{"query_id": q, "should_rerank": ideal[q], "relative_gap": gaps[q], "s0_ndcg@10": s0_n[q],
                  "s1_ndcg@10": s1_n[q], **{f"{e}_rerank": pass1[e][q]["rerank"] for e in EFFORTS}} for q in queries]
    return summary, per_query


def _f(value, spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value, spec)


def report(s: dict) -> str:
    def ci(c: dict | None) -> str:
        return "n/a" if c is None else f"{c['mean']:+.3f} [{c['low']:+.3f}, {c['high']:+.3f}] (n = {c['n']})"

    lines = [f"# S2b gate pilot ({s['mode']})", "",
             f"Calls: {s['calls']['completed']}/{s['calls']['scheduled']}; stopped by budget: {s['stopped_by_budget']}; "
             f"estimated spend ${s['estimated_spend_usd']:.4f}. Queries: {s['queries']}; oracle rerank rate "
             f"{_f(s['oracle_rerank_rate'])}. Outcome rules: {s['outcome_rules']}.", "",
             f"- Primary, S2b-medium − S2 (matched rate), share of right decisions: {ci(s['primary_medium_minus_s2_correct'])}",
             f"- S2b-low − S2 (matched): {ci(s['gates']['low']['gate_minus_s2_correct'])}",
             f"- low − medium, right decisions: {ci(s['low_minus_medium_correct'])}", "",
             "| gate | rerank rate | right decisions | S2 right (matched) | precision | recall | flip rate | "
             "unparseable (pass 1) | declines | $ per call | median s |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e, g in s["gates"].items():
        lines.append(f"| {e} | {_f(g['rerank_rate'])} | {_f(g['correct_rate'])} | {_f(g['s2_correct_rate'])} | "
                     f"{_f(g['precision'])} | {_f(g['recall'])} | {_f(g['flip_rate'])} | {_f(g['unparseable_rate_pass1'])} | "
                     f"{g['declines']} | {_f(g['gate_cost_per_call_usd'], '.5f')} | {_f(g['gate_latency_median_s'], '.1f')} |")
    lines += ["", "| strategy | rerank rate | nDCG@10 | − S1 (95% CI) | share of S1 gain | $ per query | s per query |",
              "|---|---|---|---|---|---|---|"]
    for name, v in s["strategies"].items():
        lines.append(f"| {name} | {_f(v['rerank_rate'])} | {_f(v['ndcg@10'])} | {ci(v['minus_s1'])} | "
                     f"{_f(v['share_of_s1_gain'])} | {_f(v['cost_per_query_usd'], '.5f')} | {_f(v['latency_per_query_s'], '.1f')} |")
    lines += ["", s["rerank_cost_note"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--pilot-results", type=Path, default=PILOT_RESULTS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)
    summary, per_query = analyze(args.results, args.pilot_results, args.data_dir)
    (args.results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.results / "analysis.md").write_text(report(summary))
    with (args.results / "private" / "per_query.jsonl").open("w") as f:
        for row in per_query:
            f.write(json.dumps(row) + "\n")
    print(report(summary))
    return summary


if __name__ == "__main__":
    main()
