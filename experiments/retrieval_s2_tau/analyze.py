"""S2 threshold study, free part: choose tau on the tune half, confirm on the
check half (NOT BLIND: the check-half S2 curve was seen before this contract).

    venv/bin/python experiments/retrieval_s2_tau/analyze.py

Reads, read-only: the S1 pilot's pass-1 S1-plain records (100 tune queries),
this study's records (the other 305 tune queries) and the S1 check study's
pass-1 records (404 check queries); in paid mode the reused files must match
their pinned SHA-256. Then, as the approved contract says:

1. tau* = analysis.fewest_reranks_within on all tune queries, tolerance 0.01;
2. tau* unchanged on check: H-S2 = paired bootstrap CI of S2(tau*) - S1 with
   a non-inferiority margin of -0.02, rerank rate, calls saved, cost and
   latency per query (check-half per-query rerank costs), S2 - S0, share of
   S1's gain kept, the oracle gate, and context rows for tau matched on tune
   to rates 0.5 / 0.6 / 0.8;
3. the same with safety-declined check queries excluded;
4. the section-8 outcome rules, applied mechanically.

A declined or failed rerank call keeps the BM25 order (as in every S1 study).
Writes results/summary.json and results/analysis.md (aggregates only) and
results/private/per_query.jsonl (label-derived, gitignored).
"""

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from oarp.envs.retrieval import study  # noqa: E402
from oarp.envs.retrieval.analysis import fewest_reranks_within, rank1_relevant  # noqa: E402
from oarp.envs.retrieval.bm25 import BM25Config, BM25Index  # noqa: E402
from oarp.envs.retrieval.data import DEFAULT_DATA_DIR, dev_partition, load_scifact  # noqa: E402
from oarp.envs.retrieval.metrics import ndcg_at_k  # noqa: E402
from oarp.envs.retrieval.stats import paired_bootstrap_ci  # noqa: E402
from oarp.envs.retrieval.strategies import bm25_signals, rule_gate, tau_for_rate  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PILOT_RESULTS = REPO / "experiments" / "retrieval_s1_pilot" / "results"
CHECK_RESULTS = REPO / "experiments" / "retrieval_s1_check" / "results"
REUSED_FILES = ("manifest.json", "private/sample.json", "private/calls.jsonl")
PINNED = {
    "pilot": {"manifest.json": "a8013b6d368c50ad98a96b2ba22d792431e1dc3ec19e78658329e23e7ec45cec",
              "private/sample.json": "771a2020c58ef37aa59b258ca351eb7c7a0f382b71676c8be86f8e9a12b48021",
              "private/calls.jsonl": "4a143981f208ea8fefc5fa31ad53d6fa370f16c468518b7c63bdca1714acc50e"},
    "check": {"manifest.json": "d0905eb86fbf3d973aa3bab69df0a3531327aa97f763ce01c9400539637c0220",
              "private/sample.json": "e13ab3132ea07c933c62fdfc0ea718aa7fb26bcb369d8673b260f43501fe6238",
              "private/calls.jsonl": "f0ff07cba7194c208848fbd6036772f13f41ded5da342fcfb068378cf064cb19"},
}
BOOTSTRAP_SEED = 20261024
DECLINE = "ModelRefused"
TOLERANCE = 0.01
MARGIN = -0.02
MAX_OTHER_ERROR_RATE = 0.05
DECLINE_WARNING_RATE = 0.10
CONTEXT_RATES = (0.5, 0.6, 0.8)
BLINDNESS = "NOT BLIND: the check-half S2 curve was computed and seen before this study's contract"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _mean(values) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def verify_reused(name: str, folder: Path, *, paid: bool) -> dict[str, str]:
    missing = [f for f in REUSED_FILES if not (folder / f).is_file()]
    if missing:
        raise SystemExit(f"refusing: {name} files missing: {missing}")
    hashes = {f: study.sha256_file(folder / f) for f in REUSED_FILES}
    if paid:
        changed = [f for f in REUSED_FILES if hashes[f] != PINNED[name][f]]
        if changed:
            raise SystemExit(f"refusing: {name} files differ from the pinned SHA-256: {changed}")
    return hashes


def classify(*, complete: bool, paid: bool, other_error_rate: float, decline_rate: float, tau_star: float,
             h_s2: dict | None) -> dict[str, object]:
    """Section 8 of the contract, applied mechanically."""
    if not paid:
        run = "dry run: no outcome"
    elif not complete:
        run = "incomplete (no result claimed)"
    elif other_error_rate > MAX_OTHER_ERROR_RATE:
        run = "invalid (non-decline model errors above 5%)"
    else:
        run = "valid"
    no_saving = math.isinf(tau_star) and tau_star > 0
    if run != "valid":
        h = None
    elif no_saving:
        h = "no saving possible"
    elif h_s2 is not None and h_s2["low"] > MARGIN:
        h = "non-inferior (not blind)"
    else:
        h = "not shown"
    return {"run": run, "H_S2": h, "no_saving_possible": no_saving,
            "decline_warning": decline_rate > DECLINE_WARNING_RATE, "blindness": BLINDNESS}


def analyze(results: Path, pilot_results: Path, check_results: Path, data_dir: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((results / "manifest.json").read_text())
    paid = manifest["mode"] == "paid"
    reused = {"pilot": verify_reused("pilot", pilot_results, paid=paid),
              "check": verify_reused("check", check_results, paid=paid)}
    own = _rows(results / "private" / "calls.jsonl")
    pilot_rows = [r for r in _rows(pilot_results / "private" / "calls.jsonl")
                  if r["pass"] == 1 and r["condition"] == "plain"]
    check_rows = [r for r in _rows(check_results / "private" / "calls.jsonl") if r["pass"] == 1]
    split = load_scifact(data_dir, "train")
    qrels = split.qrels
    halves = dev_partition(sorted(split.queries))

    tune_rec = {r["query_id"]: r for r in pilot_rows + own}
    check_rec = {r["query_id"]: r for r in check_rows}
    complete = (manifest["completed_calls"] == manifest["scheduled_calls"] and not manifest["stopped_by_budget"])
    if complete and set(tune_rec) != set(halves["tune"]):
        raise SystemExit("refusing: tune records do not cover exactly the tune half")
    if set(check_rec) != set(halves["check"]):
        raise SystemExit("refusing: check records do not cover exactly the check half")

    index = BM25Index(split.corpus, BM25Config())
    gaps = {}
    for q, r in {**tune_rec, **check_rec}.items():
        scored = index.rank(split.queries[q], len(r["bm25_top"]))
        if [d for d, _ in scored] != r["bm25_top"]:
            raise SystemExit("refusing: a recorded BM25 top 20 differs from a fresh BM25 ranking")
        gaps[q] = bm25_signals(scored)["relative_gap"]

    def ndcgs(recs: dict) -> tuple[dict, dict]:
        return ({q: ndcg_at_k(r["bm25_top"], qrels[q], 10) for q, r in recs.items()},
                {q: ndcg_at_k(r["ranking_top"], qrels[q], 10) for q, r in recs.items()})

    t0, t1 = ndcgs(tune_rec)
    c0, c1 = ndcgs(check_rec)
    tune_gaps = {q: gaps[q] for q in tune_rec}
    selection = fewest_reranks_within(tune_gaps, t0, t1, TOLERANCE)
    tau = selection["tau"]

    def ci(values: list[float]) -> dict | None:
        return paired_bootstrap_ci(values, seed=BOOTSTRAP_SEED) if values else None

    def gated(flags: dict, s0: dict, s1: dict) -> dict:
        return {q: s1[q] if flags[q] else s0[q] for q in flags}

    tune_flags = rule_gate(tune_gaps, tau)
    tune_s2 = gated(tune_flags, t0, t1)
    known = [r["estimated_usd"] for r in check_rec.values() if r["estimated_usd"] is not None]
    mean_cost = _mean(known) or 0.0
    cost = {q: r["estimated_usd"] if r["estimated_usd"] is not None else mean_cost for q, r in check_rec.items()}
    latency = {q: r["latency_s"] for q, r in check_rec.items()}
    declined = {q for q, r in check_rec.items() if r["error_kind"] == DECLINE}

    def confirm(t: float, ids: list[str]) -> dict:
        flags = rule_gate({q: gaps[q] for q in ids}, t)
        s2 = gated(flags, c0, c1)
        s0_mean, s1_mean, s2_mean = (_mean(m[q] for q in ids) for m in (c0, c1, s2))
        oracle = {q: not rank1_relevant(check_rec[q]["bm25_top"], qrels[q]) for q in ids}
        return {"queries": len(ids), "tau": t, "rerank_rate": _mean(flags.values()),
                "calls_saved": 1 - (_mean(flags.values()) or 0), "s0": s0_mean, "s1": s1_mean, "s2": s2_mean,
                "s2_minus_s1": ci([s2[q] - c1[q] for q in ids]), "s2_minus_s0": ci([s2[q] - c0[q] for q in ids]),
                "share_of_s1_gain_kept": (s2_mean - s0_mean) / (s1_mean - s0_mean)
                if ids and s1_mean != s0_mean else None,
                "cost_per_query_usd": _mean(cost[q] if flags[q] else 0.0 for q in ids),
                "s1_cost_per_query_usd": _mean(cost[q] for q in ids),
                "latency_per_query_s": _mean(latency[q] if flags[q] else 0.0 for q in ids),
                "s1_latency_per_query_s": _mean(latency[q] for q in ids),
                "oracle_gate": {"rerank_rate": _mean(oracle.values()),
                                "ndcg@10": _mean(gated(oracle, c0, c1).values())}}

    check_ids = sorted(check_rec)
    confirmation = confirm(tau, check_ids)
    context = []
    for rate in CONTEXT_RATES:
        t = tau_for_rate(list(tune_gaps.values()), rate)
        context.append({"tune_target_rate": rate, **{k: v for k, v in confirm(t, check_ids).items()
                                                      if k in ("tau", "rerank_rate", "s2", "s2_minus_s1")}})
    curve = [c for c in selection["curve"] if any(abs(c["rate"] - r / 20) < 1 / (2 * len(tune_gaps)) for r in range(21))]
    errors = Counter(r["error_kind"] for r in own if r["status"] == "model_error")
    summary = {
        "study": manifest["study"], "mode": manifest["mode"], "git_at_run": manifest["git"],
        "calls": {"scheduled": manifest["scheduled_calls"], "completed": manifest["completed_calls"]},
        "stopped_by_budget": manifest["stopped_by_budget"], "estimated_spend_usd": manifest["estimated_spend_usd"],
        "reused_sha256": reused, "blindness": BLINDNESS,
        "tune": {"queries": len(tune_rec), "from_pilot": len(pilot_rows), "from_this_study": len(own),
                 "tolerance": TOLERANCE, "tau_star": tau, "rerank_rate": selection["rate"],
                 "s0": _mean(t0.values()), "s1": _mean(t1.values()), "s2": _mean(tune_s2.values()),
                 "s2_minus_s1": ci([tune_s2[q] - t1[q] for q in tune_rec]),
                 "curve_every_5_percent": curve},
        "check_confirmation": {"margin": MARGIN, **confirmation},
        "check_excluding_declined": confirm(tau, [q for q in check_ids if q not in declined]),
        "check_declined_queries": len(declined),
        "context_fixed_rates": context,
        "this_study_calls": {"status_counts": dict(Counter(r["status"] for r in own)),
                             "declines": errors.get(DECLINE, 0),
                             "decline_rate": errors.get(DECLINE, 0) / len(own) if own else 0.0,
                             "other_error_rate": (sum(errors.values()) - errors.get(DECLINE, 0)) / len(own) if own else 0.0,
                             "estimated_usd": sum(r["estimated_usd"] or 0.0 for r in own),
                             "mean_input_tokens": _mean(r["usage"]["input_tokens"] for r in own if r["usage"]),
                             "mean_output_tokens": _mean(r["usage"]["output_tokens"] for r in own if r["usage"])},
        "cost_note": f"check-half per-query rerank costs; declined calls at the check mean ${mean_cost:.5f}",
    }
    summary["outcome_rules"] = classify(
        complete=complete, paid=paid, other_error_rate=summary["this_study_calls"]["other_error_rate"],
        decline_rate=summary["this_study_calls"]["decline_rate"], tau_star=tau,
        h_s2=confirmation["s2_minus_s1"])
    per_query = [{"query_id": q, "half": "tune" if q in tune_rec else "check", "relative_gap": gaps[q],
                  "s0_ndcg@10": (t0 | c0)[q], "s1_ndcg@10": (t1 | c1)[q]} for q in sorted(gaps)]
    return summary, per_query


def _f(value, spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value, spec)


def report(s: dict) -> str:
    def ci(c: dict | None) -> str:
        return "n/a" if c is None else f"{c['mean']:+.3f} [{c['low']:+.3f}, {c['high']:+.3f}] (n = {c['n']})"

    t, c = s["tune"], s["check_confirmation"]
    lines = [
        f"# S2 threshold study ({s['mode']})", "", f"**{s['blindness']}.**", "",
        f"Calls: {s['calls']['completed']}/{s['calls']['scheduled']}; stopped by budget: {s['stopped_by_budget']}; "
        f"estimated spend ${s['estimated_spend_usd']:.4f}. Outcome rules: {s['outcome_rules']}.", "",
        f"## Selection on tune ({t['queries']} queries, tolerance {t['tolerance']})", "",
        f"tau* = {t['tau_star']}, rerank rate {_f(t['rerank_rate'])}; nDCG@10 S0 {_f(t['s0'])}, S1 {_f(t['s1'])}, "
        f"S2 {_f(t['s2'])}; S2 − S1 {ci(t['s2_minus_s1'])}.", "",
        f"## Confirmation on check (margin {c['margin']}; not blind)", "",
        f"- S2(tau*) − S1: {ci(c['s2_minus_s1'])}",
        f"- S2(tau*) − S0: {ci(c['s2_minus_s0'])}",
        f"- rerank rate {_f(c['rerank_rate'])} (calls saved {_f(c['calls_saved'])}); nDCG@10 S0 {_f(c['s0'])}, "
        f"S1 {_f(c['s1'])}, S2 {_f(c['s2'])}; share of S1 gain kept {_f(c['share_of_s1_gain_kept'])}",
        f"- $ per query {_f(c['cost_per_query_usd'], '.5f')} vs always rerank {_f(c['s1_cost_per_query_usd'], '.5f')}; "
        f"s per query {_f(c['latency_per_query_s'], '.2f')} vs {_f(c['s1_latency_per_query_s'], '.2f')}",
        f"- oracle gate: {c['oracle_gate']}",
        f"- excluding {s['check_declined_queries']} declined queries: "
        f"{ci(s['check_excluding_declined']['s2_minus_s1'])}", "",
        "## Context: tau matched on tune to fixed rates, applied to check", "",
        "| tune target rate | tau | check rate | check S2 nDCG@10 | S2 − S1 |", "|---|---|---|---|---|",
        *[f"| {r['tune_target_rate']} | {_f(r['tau'], '.4f')} | {_f(r['rerank_rate'])} | {_f(r['s2'])} | "
          f"{ci(r['s2_minus_s1'])} |" for r in s["context_fixed_rates"]], "",
        f"This study's calls: {s['this_study_calls']}. {s['cost_note']}.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--pilot-results", type=Path, default=PILOT_RESULTS)
    parser.add_argument("--check-results", type=Path, default=CHECK_RESULTS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)
    summary, per_query = analyze(args.results, args.pilot_results, args.check_results, args.data_dir)
    (args.results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.results / "analysis.md").write_text(report(summary))
    with (args.results / "private" / "per_query.jsonl").open("w") as f:
        for row in per_query:
            f.write(json.dumps(row) + "\n")
    print(report(summary))
    return summary


if __name__ == "__main__":
    main()
