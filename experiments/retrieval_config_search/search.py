"""Automated configuration search over retrieval harness policies (offline, $0).

    venv/bin/python experiments/retrieval_config_search/search.py   # free: no model calls

The space (bounded, label-free, fixed before any score is computed): a policy
routes each query, using BM25's own scores, to one of three recorded
pipelines:

    if search_signal < tau_search:   S3  (search agent + forced rerank)
    elif rerank_signal < tau_rerank: S1  (rerank BM25's top 20)
    else:                            S0  (BM25's order)

- signals: relative gap, raw gap, top score (`strategies.bm25_signals`), each
  for both decisions;
- rates: 0, 0.1, ..., 1.0 per decision; tau = `strategies.tau_for_rate` over
  the signal on all 809 development queries (label-free quantiles). The tau
  VALUES are what a policy is, so a selected policy transfers unchanged;
- 3 x 11 x 3 x 11 = 1,089 policies (some coincide, e.g. search rate 1.0).

Selection set: the 404 check queries, where S0, S1 and S3 are all recorded
(development data; the check half was spent as S3's confirmation, so selecting
on it is allowed, and nothing selected here may be confirmed on it).

Pre-registered selection rule: best = the highest mean nDCG@10 on check; the
SELECTED policy = the cheapest (mean estimated $ per query) policy within
0.005 of best; ties to higher nDCG@10, then lower search rate, then lower
rerank rate, then signal order. Also reported: the cost/quality Pareto
frontier on check, and the selected policy on the stratified tune sample
(163 queries, tune weights) as a descriptive cross-check.

Costs: recorded per-query estimates; S1 calls without a price (declines) are
priced at the mean of the priced S1 calls; S3 calls without a price count as $0
(counted). S0 costs $0. Writes results/summary.json, results/analysis.md
(aggregates only) and results/private/per_query.jsonl.
"""

import argparse
import itertools
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from oarp.envs.retrieval import study  # noqa: E402
from oarp.envs.retrieval.bm25 import BM25Config, BM25Index  # noqa: E402
from oarp.envs.retrieval.data import DEFAULT_DATA_DIR, dev_partition, load_scifact  # noqa: E402
from oarp.envs.retrieval.metrics import ndcg_at_k  # noqa: E402
from oarp.envs.retrieval.stats import paired_bootstrap_ci  # noqa: E402
from oarp.envs.retrieval.strategies import bm25_signals, tau_for_rate  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
EXP = REPO / "experiments"
INPUTS = {
    "s1_check": EXP / "retrieval_s1_check" / "results" / "private" / "calls.jsonl",
    "s3_check": EXP / "retrieval_s3_check" / "results" / "private" / "episodes.jsonl",
    "s1_pilot": EXP / "retrieval_s1_pilot" / "results" / "private" / "calls.jsonl",
    "s1_tau": EXP / "retrieval_s2_tau" / "results" / "private" / "calls.jsonl",
    "s3_pilot": EXP / "retrieval_s3_pilot" / "results" / "private" / "episodes.jsonl",
    "s3_pilot_manifest": EXP / "retrieval_s3_pilot" / "results" / "manifest.json",
}
PINNED = {
    "s1_check": "f0ff07cba7194c208848fbd6036772f13f41ded5da342fcfb068378cf064cb19",
    "s3_check": "564d218c3cf865c69bfe2716d9e3613eaf182d62c21254624b4ef79f5c7469f9",
    "s1_pilot": "4a143981f208ea8fefc5fa31ad53d6fa370f16c468518b7c63bdca1714acc50e",
    "s1_tau": "7395477e2c05603c39b14f170445977015589d9d3b0d48060e5f4859dd37a0c0",
    "s3_pilot": "4f41bc759b8aad91752b696de57e4f8e52d92a1786692c01d289997ac8c94fbf",
    "s3_pilot_manifest": "d810b4fa9fd3a909a0f7c32c1195e578eba54fa573fcd974b327dc5994d5631b",
}
SIGNALS = ("relative_gap", "raw_gap", "top_score")
RATES = tuple(round(0.1 * i, 1) for i in range(11))
TOLERANCE = 0.005
BOOTSTRAP_SEED = 20261029
DEPTH = 20


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def verify_inputs(inputs: dict[str, Path], *, pinned: bool) -> dict[str, str]:
    missing = [n for n, p in inputs.items() if not p.is_file()]
    if missing:
        raise SystemExit(f"refusing: input files missing: {missing}")
    hashes = {n: study.sha256_file(p) for n, p in inputs.items()}
    if pinned:
        changed = [n for n in inputs if hashes[n] != PINNED[n]]
        if changed:
            raise SystemExit(f"refusing: inputs differ from their pinned SHA-256: {changed}")
    return hashes


def route(row: dict, policy: dict) -> str:
    """The pipeline a policy sends this query to."""
    if row[policy["search_signal"]] < policy["search_tau"]:
        return "s3"
    if row[policy["rerank_signal"]] < policy["rerank_tau"]:
        return "s1"
    return "s0"


def score(rows: list[dict], policy: dict, weights: dict[str, float] | None = None) -> dict[str, float]:
    """Mean nDCG@10, $ and route shares; reweighted to population shares when `weights` is given."""
    def mean(value) -> float:
        if weights is None:
            return statistics.fmean(value(r) for r in rows)
        total = sum(weights.values())
        return sum(w / total * statistics.fmean(value(r) for r in rows if r["stratum"] == s) for s, w in weights.items())

    return {"ndcg": mean(lambda r: r[route(r, policy)]), "usd": mean(lambda r: r[f"{route(r, policy)}_usd"]),
            "s": mean(lambda r: r[f"{route(r, policy)}_s"]),
            "share_s3": mean(lambda r: route(r, policy) == "s3"), "share_s1": mean(lambda r: route(r, policy) == "s1")}


def select(results: list[dict], tolerance: float = TOLERANCE) -> tuple[dict, dict]:
    """(best, selected): the cheapest policy within `tolerance` of the best nDCG@10."""
    best = max(results, key=lambda r: r["ndcg"])
    near = [r for r in results if r["ndcg"] >= best["ndcg"] - tolerance - 1e-12]
    selected = min(near, key=lambda r: (r["usd"], -r["ndcg"], r["search_rate"], r["rerank_rate"],
                                         SIGNALS.index(r["search_signal"]), SIGNALS.index(r["rerank_signal"])))
    return best, selected


def pareto(results: list[dict]) -> list[dict]:
    """Policies no other policy beats on both cost (lower) and nDCG@10 (higher), by cost."""
    frontier, best = [], -math.inf
    for r in sorted(results, key=lambda r: (r["usd"], -r["ndcg"])):
        if r["ndcg"] > best + 1e-12:
            frontier.append(r)
            best = r["ndcg"]
    return frontier


def space(signal_values: dict[str, list[float]]) -> list[dict]:
    """All policies, with tau values from label-free quantiles over the development queries."""
    taus = {sig: {rate: tau_for_rate(vals, rate) for rate in RATES} for sig, vals in signal_values.items()}
    return [{"search_signal": ss, "search_rate": sr, "search_tau": taus[ss][sr],
             "rerank_signal": rs, "rerank_rate": rr, "rerank_tau": taus[rs][rr]}
            for ss, sr, rs, rr in itertools.product(SIGNALS, RATES, SIGNALS, RATES)]


def analyze(data_dir: Path, inputs: dict[str, Path] = INPUTS, *, pinned: bool = True) -> tuple[dict, list[dict]]:
    hashes = verify_inputs(inputs, pinned=pinned)
    split = load_scifact(data_dir, "train")  # development only; the test split stays unloaded
    qrels = split.qrels
    halves = dev_partition(sorted(split.queries))
    dev = sorted(halves["tune"] + halves["check"])
    index = BM25Index(split.corpus, BM25Config())
    ranked = {q: index.rank(split.queries[q]) for q in dev}
    sig = {q: bm25_signals(ranked[q]) for q in dev}
    signal_values = {s: [sig[q][s] for q in dev] for s in SIGNALS}

    s1_check = {r["query_id"]: r for r in _rows(inputs["s1_check"]) if r["pass"] == 1}
    s1_tune = {r["query_id"]: r for r in _rows(inputs["s1_pilot"]) if r["pass"] == 1 and r["condition"] == "plain"}
    s1_tune |= {r["query_id"]: r for r in _rows(inputs["s1_tau"])}
    s3_check = {r["query_id"]: r for r in _rows(inputs["s3_check"]) if r["phase"] == "main"}
    s3_tune = {r["query_id"]: r for r in _rows(inputs["s3_pilot"]) if r["phase"] == "main"}
    strata = json.loads(inputs["s3_pilot_manifest"].read_text())["strata_tune"]
    weights = {"beyond": strata["beyond"], "within": strata["within_available"]}
    if set(s3_check) != set(halves["check"]) or set(s1_check) != set(halves["check"]):
        raise SystemExit("refusing: the check records do not cover exactly the check half")
    priced = [r["estimated_usd"] for r in list(s1_check.values()) + list(s1_tune.values())
              if r["estimated_usd"] is not None]
    s1_fill = statistics.fmean(priced)

    def make_row(q: str, s1: dict, s3: dict) -> dict:
        bm25 = [d for d, _ in ranked[q]]
        if s1["bm25_top"] != bm25[:DEPTH] or s3["candidates"][:DEPTH] != bm25[:DEPTH]:
            raise SystemExit("refusing: a recorded BM25 top 20 differs from a fresh BM25 ranking")
        s3_calls = s3["call_usd"]["agent"] + s3["call_usd"]["rerank"]
        return {"query_id": q, "stratum": s3["stratum"], **sig[q],
                "s0": ndcg_at_k(bm25, qrels[q], 10), "s0_usd": 0.0, "s0_s": 0.0,
                "s1": ndcg_at_k(s1["ranking_top"] + bm25[DEPTH:], qrels[q], 10),
                "s1_usd": s1["estimated_usd"] if s1["estimated_usd"] is not None else s1_fill,
                "s1_s": s1.get("latency_s") or 0.0, "s1_unpriced": s1["estimated_usd"] is None,
                "s3": ndcg_at_k(s3["ranking"], qrels[q], 10), "s3_usd": sum(c for c in s3_calls if c is not None),
                "s3_s": s3["wall_s"], "s3_unpriced": sum(c is None for c in s3_calls)}

    check_rows = [make_row(q, s1_check[q], s3_check[q]) for q in sorted(halves["check"])]
    tune_rows = [make_row(q, s1_tune[q], s3_tune[q]) for q in sorted(s3_tune)]

    results = []
    for policy in space(signal_values):
        results.append({**policy, **score(check_rows, policy)})
    best, selected = select(results)
    frontier = pareto(results)
    fixed = {name: next(r for r in results if r["search_rate"] == sr and r["rerank_rate"] == rr
                        and r["search_signal"] == "relative_gap" and r["rerank_signal"] == "relative_gap")
             for name, sr, rr in (("S0", 0.0, 0.0), ("S1", 0.0, 1.0), ("S3", 1.0, 0.0))}

    def ci_vs(policy: dict, other: str) -> dict:
        return paired_bootstrap_ci([r[route(r, policy)] - r[other] for r in check_rows], seed=BOOTSTRAP_SEED)

    tune_selected = score(tune_rows, selected, weights)
    summary = {
        "study": "Automated configuration search over BM25-routed policies (S0 / S1 / S3), offline, $0",
        "git": study.preflight(REPO, RESULTS, data_dir, None, paid=False)["git"],
        "inputs_sha256": hashes, "pinned": pinned, "signals": SIGNALS, "rates": RATES, "tolerance": TOLERANCE,
        "policies": len(results), "distinct_outcomes": len({(round(r["ndcg"], 12), round(r["usd"], 12)) for r in results}),
        "queries": {"selection_check": len(check_rows), "thresholds_dev": len(dev), "cross_check_tune": len(tune_rows)},
        "s1_unpriced_filled_usd": s1_fill,
        "unpriced": {"s1_check": sum(r["s1_unpriced"] for r in check_rows),
                     "s3_check_calls": sum(r["s3_unpriced"] for r in check_rows)},
        "fixed_pipelines_check": fixed, "best_check": best, "selected": selected,
        "selected_vs_check": {"minus_s1": ci_vs(selected, "s1"), "minus_s3": ci_vs(selected, "s3"),
                              "minus_s0": ci_vs(selected, "s0")},
        "selected_tune_cross_check": {"selected": tune_selected,
                                      "s1": score(tune_rows, fixed["S1"], weights),
                                      "s3": score(tune_rows, fixed["S3"], weights),
                                      "note": "descriptive: stratified tune sample, tune weights"},
        "pareto_frontier_check": frontier,
        "note": "selection data: the selected policy's check numbers are optimistic; only the test split can validate it",
    }
    per_query = [{**r, "half": "check"} for r in check_rows] + [{**r, "half": "tune"} for r in tune_rows]
    return summary, per_query


def _f(value, spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value, spec)


def describe(p: dict) -> str:
    return (f"search if {p['search_signal']} < {_f(p['search_tau'], '.4f')} (rate {p['search_rate']}), else rerank "
            f"if {p['rerank_signal']} < {_f(p['rerank_tau'], '.4f')} (rate {p['rerank_rate']}), else BM25")


def report(s: dict) -> str:
    def ci(c: dict) -> str:
        return f"{c['mean']:+.3f} [{c['low']:+.3f}, {c['high']:+.3f}]"

    sel, best, fx = s["selected"], s["best_check"], s["fixed_pipelines_check"]
    lines = [
        "# Configuration search over BM25-routed retrieval policies (offline, $0)", "",
        f"{s['policies']} policies ({s['distinct_outcomes']} distinct outcomes) scored on {s['queries']['selection_check']} "
        f"check queries; thresholds from {s['queries']['thresholds_dev']} development queries. Selection data: the "
        "selected policy's numbers here are optimistic.", "",
        "| policy | nDCG@10 | $ per query | s per query | share S3 | share S1 |", "|---|---|---|---|---|---|",
        *[f"| {n} | {_f(p['ndcg'])} | {_f(p['usd'], '.5f')} | {_f(p['s'], '.1f')} | {_f(p['share_s3'], '.2f')} | "
          f"{_f(p['share_s1'], '.2f')} |" for n, p in (*fx.items(), ("best", best), ("**selected**", sel))], "",
        f"Selected (cheapest within {s['tolerance']} of best): {describe(sel)}.",
        f"- selected − S1 {ci(s['selected_vs_check']['minus_s1'])}; − S3 {ci(s['selected_vs_check']['minus_s3'])}; "
        f"− S0 {ci(s['selected_vs_check']['minus_s0'])} (check, optimistic)",
        f"- tune cross-check (stratified, descriptive): selected {_f(s['selected_tune_cross_check']['selected']['ndcg'])} "
        f"at ${_f(s['selected_tune_cross_check']['selected']['usd'], '.5f')}; S1 "
        f"{_f(s['selected_tune_cross_check']['s1']['ndcg'])}; S3 {_f(s['selected_tune_cross_check']['s3']['ndcg'])}", "",
        "## Pareto frontier on check (cost ascending)", "", "| $ per query | nDCG@10 | share S3 | share S1 | policy |",
        "|---|---|---|---|---|",
        *[f"| {_f(p['usd'], '.5f')} | {_f(p['ndcg'])} | {_f(p['share_s3'], '.2f')} | {_f(p['share_s1'], '.2f')} | "
          f"{describe(p)} |" for p in s["pareto_frontier_check"]], "",
        f"Unpriced: {s['unpriced']}; S1 declines priced at ${s['s1_unpriced_filled_usd']:.5f}.",
    ]
    return "\n".join(lines) + "\n"


def _clean(value):
    if isinstance(value, float) and math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def write(results: Path, summary: dict, rows: list[dict]) -> None:
    (results / "private").mkdir(parents=True, exist_ok=True)
    (results / "summary.json").write_text(json.dumps(_clean(summary), indent=2) + "\n")
    (results / "analysis.md").write_text(report(summary))
    with (results / "private" / "per_query.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    summary, rows = analyze(args.data_dir)
    write(args.results, summary, rows)
    print(report(summary))
    return summary


if __name__ == "__main__":
    main()
