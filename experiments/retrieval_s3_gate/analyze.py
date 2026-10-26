"""Gate study (free, offline): run the extra search only when BM25 looks unsure.

    venv/bin/python experiments/retrieval_s3_gate/analyze.py   # $0: no model calls

A gate chooses, per query, between two outcomes that already exist: S1 (BM25's
top 20 reranked, no extra search) and a search pipeline's outcome (S3 = the
agent + forced rerank, or R1 = one reformulated search + forced rerank).
R1 without an extra search is S1 by construction, so gated systems are scored
from stored records, as S2 was. Search iff signal < tau ("unsure").

Part A (tune, selection data): the 163 stratified tune queries of the S3 and
R1 pilots. Candidate gates, fixed in advance and label-free: 3 BM25 signals
(relative gap, raw gap, top score) x 9 search rates (0.2 .. 1.0), tau =
`strategies.tau_for_rate` over the signal on all 405 tune queries. Selection
rule per base system (G-S3, G-R1): highest stratified mean nDCG@10 (tune
weights 63 : 342); ties to the lower achieved search rate, then to signal
order. Selected numbers are optimistic (chosen on the same queries).

Part B (check half, NOT BLIND: S3's check results were known before): the
selected G-S3 gate (signal and tau value unchanged) on all 404 check queries,
from the S3 check episodes and the S1 check records. D1: G-S3 - S1 > 0 (CI
lower bound); D2: within-20 G-S3 - S1 > -0.02. Labels: "gate removes the harm
(not blind)" (D1 and D2), "gate does not help" (D1, not D2), "gate loses the
gain" (not D1).

Inputs are pinned by SHA-256. Writes results/summary.json and
results/analysis.md (aggregates only) and results/private/per_query.jsonl.
"""

import argparse
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
from oarp.envs.retrieval.stats import paired_bootstrap_ci, stratified_bootstrap_ci  # noqa: E402
from oarp.envs.retrieval.strategies import bm25_signals, tau_for_rate  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
EXP = REPO / "experiments"
INPUTS = {
    "s1_pilot": EXP / "retrieval_s1_pilot" / "results" / "private" / "calls.jsonl",
    "s1_tau": EXP / "retrieval_s2_tau" / "results" / "private" / "calls.jsonl",
    "s1_check": EXP / "retrieval_s1_check" / "results" / "private" / "calls.jsonl",
    "s3_pilot": EXP / "retrieval_s3_pilot" / "results" / "private" / "episodes.jsonl",
    "s3_pilot_manifest": EXP / "retrieval_s3_pilot" / "results" / "manifest.json",
    "r1_pilot": EXP / "retrieval_s3_r1_pilot" / "results" / "private" / "episodes.jsonl",
    "s3_check": EXP / "retrieval_s3_check" / "results" / "private" / "episodes.jsonl",
}
PINNED = {
    "s1_pilot": "4a143981f208ea8fefc5fa31ad53d6fa370f16c468518b7c63bdca1714acc50e",
    "s1_tau": "7395477e2c05603c39b14f170445977015589d9d3b0d48060e5f4859dd37a0c0",
    "s1_check": "f0ff07cba7194c208848fbd6036772f13f41ded5da342fcfb068378cf064cb19",
    "s3_pilot": "4f41bc759b8aad91752b696de57e4f8e52d92a1786692c01d289997ac8c94fbf",
    "s3_pilot_manifest": "d810b4fa9fd3a909a0f7c32c1195e578eba54fa573fcd974b327dc5994d5631b",
    "r1_pilot": "b77e5a2889e3b697bbe187d3892d3235b8e58a2b2f99ce5f803ab9c5e0b9505d",
    "s3_check": "564d218c3cf865c69bfe2716d9e3613eaf182d62c21254624b4ef79f5c7469f9",
}
SIGNALS = ("relative_gap", "raw_gap", "top_score")  # also the tie-break order
RATES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
BASES = ("s3", "r1")
BOOTSTRAP_SEED = 20261028
MARGIN = -0.02
DEPTH = 20
NOT_BLIND = "NOT BLIND: S3's check results (overall and per stratum) were known before this analysis"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _mean(values) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


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


def gated(row: dict, base: str, signal: str, tau: float) -> dict[str, float]:
    """The gated system's outcome for one query: the base pipeline if unsure, else S1."""
    search = row[signal] < tau
    pick = base if search else "s1"
    return {"ndcg": row[pick], "usd": row[f"{pick}_usd"], "s": row[f"{pick}_s"], "searched": float(search)}


def weighted(rows: list[dict], weights: dict[str, float], value) -> float:
    """Stratum means reweighted to the population shares."""
    total = sum(weights.values())
    return sum(weights[s] / total * statistics.fmean(value(r) for r in rows if r["stratum"] == s) for s in weights)


def select(curve: list[dict]) -> dict:
    """Highest stratified nDCG@10; ties to the lower achieved search rate, then signal order."""
    return max(curve, key=lambda c: (round(c["ndcg"], 12), -c["achieved_rate_tune"], -SIGNALS.index(c["signal"])))


def check_outcome(d1: dict | None, d2: dict | None) -> str | None:
    if d1 is None or d2 is None:
        return None
    if d1["low"] <= 0:
        return "gate loses the gain"
    return "gate removes the harm (not blind)" if d2["low"] > MARGIN else "gate does not help"


def analyze(data_dir: Path, inputs: dict[str, Path] = INPUTS, *, pinned: bool = True) -> tuple[dict, list[dict]]:
    hashes = verify_inputs(inputs, pinned=pinned)
    split = load_scifact(data_dir, "train")
    qrels = split.qrels
    halves = dev_partition(sorted(split.queries))
    index = BM25Index(split.corpus, BM25Config())
    s1 = {r["query_id"]: r for r in _rows(inputs["s1_pilot"]) if r["pass"] == 1 and r["condition"] == "plain"}
    s1 |= {r["query_id"]: r for r in _rows(inputs["s1_tau"])}
    s1_check = {r["query_id"]: r for r in _rows(inputs["s1_check"]) if r["pass"] == 1}
    s3 = {r["query_id"]: r for r in _rows(inputs["s3_pilot"]) if r["phase"] == "main"}
    r1 = {r["query_id"]: r for r in _rows(inputs["r1_pilot"]) if r["phase"] == "main"}
    s3_check = {r["query_id"]: r for r in _rows(inputs["s3_check"]) if r["phase"] == "main"}
    strata = json.loads(inputs["s3_pilot_manifest"].read_text())["strata_tune"]
    weights = {"beyond": strata["beyond"], "within": strata["within_available"]}
    if set(s3) != set(r1):
        raise SystemExit("refusing: the S3 and R1 pilot episodes cover different queries")

    signals = {}
    for q in sorted(set(halves["tune"]) | set(halves["check"])):
        scored = index.rank(split.queries[q])
        signals[q] = (bm25_signals(scored), [d for d, _ in scored])

    def s1_row(q: str, rec: dict) -> dict:
        sig, bm25 = signals[q]
        if rec["bm25_top"] != bm25[:DEPTH]:
            raise SystemExit("refusing: a recorded BM25 top 20 differs from a fresh BM25 ranking")
        return {**sig, "s1": ndcg_at_k(rec["ranking_top"] + bm25[DEPTH:], qrels[q], 10),
                "s1_usd": rec.get("estimated_usd") or 0.0, "s1_s": rec.get("latency_s") or 0.0}

    def pipeline(e: dict, name: str) -> dict:
        usd = sum(c for c in e["call_usd"]["agent"] + e["call_usd"]["rerank"] if c is not None)
        return {name: ndcg_at_k(e["ranking"], qrels[e["query_id"]], 10), f"{name}_usd": usd, f"{name}_s": e["wall_s"]}

    tune_rows = [{"query_id": q, "stratum": s3[q]["stratum"], **s1_row(q, s1[q]), **pipeline(s3[q], "s3"),
                  **pipeline(r1[q], "r1")} for q in sorted(s3)]
    check_rows = [{"query_id": q, "stratum": e["stratum"], **s1_row(q, s1_check[q]), **pipeline(e, "s3")}
                  for q, e in sorted(s3_check.items())]

    # Part A: thresholds from label-free quantiles over all tune queries.
    tune_all = halves["tune"]
    curves, selected = {}, {}
    for base in BASES:
        curve = []
        for signal in SIGNALS:
            values = [signals[q][0][signal] for q in tune_all]
            for rate in RATES:
                tau = tau_for_rate(values, rate)
                achieved = statistics.fmean(v < tau for v in values)
                curve.append({
                    "signal": signal, "target_rate": rate, "tau": tau, "achieved_rate_tune": achieved,
                    "ndcg": weighted(tune_rows, weights, lambda r: gated(r, base, signal, tau)["ndcg"]),
                    "search_rate_sample": weighted(tune_rows, weights, lambda r: gated(r, base, signal, tau)["searched"]),
                    "usd": weighted(tune_rows, weights, lambda r: gated(r, base, signal, tau)["usd"]),
                    "s": weighted(tune_rows, weights, lambda r: gated(r, base, signal, tau)["s"]),
                    "within": _mean(gated(r, base, signal, tau)["ndcg"] for r in tune_rows if r["stratum"] == "within"),
                    "beyond": _mean(gated(r, base, signal, tau)["ndcg"] for r in tune_rows if r["stratum"] == "beyond"),
                })
        best = select(curve)
        sig, tau = best["signal"], best["tau"]

        def diff(rows, other, keep=lambda r: True):
            return {s: [gated(r, base, sig, tau)["ndcg"] - r[other] for r in rows if r["stratum"] == s and keep(r)]
                    for s in weights}

        parts_s1, parts_base = diff(tune_rows, "s1"), diff(tune_rows, base)
        beyond_rows = [r for r in tune_rows if r["stratum"] == "beyond"]
        gains = [r for r in beyond_rows if r[base] > r["s1"]]
        selected[base] = {
            **best,
            "gated_minus_s1": stratified_bootstrap_ci(parts_s1, weights, seed=BOOTSTRAP_SEED),
            "gated_minus_always_search": stratified_bootstrap_ci(parts_base, weights, seed=BOOTSTRAP_SEED),
            "within_gated_minus_s1": paired_bootstrap_ci(parts_s1["within"], seed=BOOTSTRAP_SEED),
            "beyond_gated_minus_s1": paired_bootstrap_ci(parts_s1["beyond"], seed=BOOTSTRAP_SEED),
            "beyond_gains_kept": {"queries_where_search_helped": len(gains),
                                  "kept_by_gate": sum(r[sig] < tau for r in gains)},
            "always_search": next(c for c in curve if c["signal"] == sig and c["target_rate"] == 1.0),
            "s1_reference": {"ndcg": weighted(tune_rows, weights, lambda r: r["s1"]),
                             "usd": weighted(tune_rows, weights, lambda r: r["s1_usd"])},
        }
        curves[base] = curve

    # Part B (not blind): the selected G-S3 gate, unchanged, on the check half.
    g = selected["s3"]
    sig, tau = g["signal"], g["tau"]
    out = [gated(r, "s3", sig, tau) for r in check_rows]
    within = [(o, r) for o, r in zip(out, check_rows) if r["stratum"] == "within"]

    def ci(pairs, other):
        return paired_bootstrap_ci([o["ndcg"] - r[other] for o, r in pairs], seed=BOOTSTRAP_SEED) if pairs else None

    pairs = list(zip(out, check_rows))
    d1, d2 = ci(pairs, "s1"), ci(within, "s1")
    part_b = {
        "blindness": NOT_BLIND, "signal": sig, "tau": tau, "queries": len(check_rows),
        "search_rate": _mean(o["searched"] for o in out),
        "means": {"gated": _mean(o["ndcg"] for o in out), "s1": _mean(r["s1"] for r in check_rows),
                  "s3": _mean(r["s3"] for r in check_rows)},
        "D1_gated_minus_s1": d1, "D2_within_gated_minus_s1": d2, "gated_minus_s3": ci(pairs, "s3"),
        "beyond_gated_minus_s1": ci([(o, r) for o, r in pairs if r["stratum"] == "beyond"], "s1"),
        "usd_per_query": {"gated": _mean(o["usd"] for o in out), "s3": _mean(r["s3_usd"] for r in check_rows),
                          "s1": _mean(r["s1_usd"] for r in check_rows)},
        "s_per_query": {"gated": _mean(o["s"] for o in out), "s3": _mean(r["s3_s"] for r in check_rows),
                        "s1": _mean(r["s1_s"] for r in check_rows)},
        "outcome": check_outcome(d1, d2),
    }
    summary = {
        "study": "Gate study: extra search only when BM25 looks unsure (offline, $0)",
        "git": study.preflight(REPO, RESULTS, data_dir, None, paid=False)["git"],
        "inputs_sha256": hashes, "pinned": pinned, "bootstrap_seed": BOOTSTRAP_SEED,
        "signals": SIGNALS, "rates": RATES, "weights_tune": weights,
        "queries": {"tune_sample": len(tune_rows), "tune_all_for_thresholds": len(tune_all), "check": len(check_rows)},
        "part_a_selected": selected, "part_a_curves": curves, "part_b_check_not_blind": part_b,
    }
    per_query = [{**r, "half": "tune"} for r in tune_rows] + [{**r, "half": "check"} for r in check_rows]
    return summary, per_query


def _f(value, spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value, spec)


def report(s: dict) -> str:
    def ci(c: dict | None) -> str:
        return "n/a" if c is None else f"{c['mean']:+.3f} [{c['low']:+.3f}, {c['high']:+.3f}]"

    lines = ["# Gate study: search only when BM25 looks unsure (offline, $0)", "",
             f"Inputs pinned: {s['pinned']}. Tune sample {s['queries']['tune_sample']} queries (weights "
             f"{s['weights_tune']}), thresholds from {s['queries']['tune_all_for_thresholds']} tune queries; check "
             f"{s['queries']['check']} queries. One recorded episode per query.", ""]
    for base, label in (("s3", "G-S3 (gate the agent)"), ("r1", "G-R1 (gate the fixed reformulation)")):
        g = s["part_a_selected"][base]
        a = g["always_search"]
        lines += [
            f"## Part A, tune (selection data): {label}", "",
            "| signal | rate 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |", "|---|" + "---|" * 9,
            *[f"| {sig} | " + " | ".join(_f(c["ndcg"]) for c in s["part_a_curves"][base] if c["signal"] == sig) + " |"
              for sig in s["signals"]], "",
            f"Selected: {g['signal']} at target rate {g['target_rate']} (tau {_f(g['tau'], '.4f')}, achieved "
            f"{_f(g['achieved_rate_tune'])} of tune, {_f(g['search_rate_sample'])} weighted in the sample). "
            f"nDCG@10 {_f(g['ndcg'])} (always search {_f(a['ndcg'])}, S1 {_f(g['s1_reference']['ndcg'])}); "
            f"within {_f(g['within'])}, beyond {_f(g['beyond'])}.",
            f"- gated − S1: {ci(g['gated_minus_s1'])}; gated − always search: {ci(g['gated_minus_always_search'])}",
            f"- within gated − S1: {ci(g['within_gated_minus_s1'])}; beyond gated − S1: "
            f"{ci(g['beyond_gated_minus_s1'])}",
            f"- beyond queries where search helped: {g['beyond_gains_kept']['queries_where_search_helped']}, kept by "
            f"the gate: {g['beyond_gains_kept']['kept_by_gate']}",
            f"- $ per query: gated {_f(g['usd'], '.5f')}, always search {_f(a['usd'], '.5f')}, S1 "
            f"{_f(g['s1_reference']['usd'], '.5f')}; s per query gated {_f(g['s'], '.1f')}, always {_f(a['s'], '.1f')}",
            "- optimistic: chosen on the same queries", ""]
    b = s["part_b_check_not_blind"]
    lines += [
        f"## Part B, check half: the selected G-S3 gate unchanged ({b['blindness']})", "",
        f"Signal {b['signal']}, tau {_f(b['tau'], '.4f')}; search rate on check {_f(b['search_rate'])}. nDCG@10: gated "
        f"{_f(b['means']['gated'])}, S1 {_f(b['means']['s1'])}, S3 {_f(b['means']['s3'])}.",
        f"- D1 gated − S1: {ci(b['D1_gated_minus_s1'])}",
        f"- D2 within gated − S1 (margin {MARGIN}): {ci(b['D2_within_gated_minus_s1'])}",
        f"- gated − S3: {ci(b['gated_minus_s3'])}; beyond gated − S1: {ci(b['beyond_gated_minus_s1'])}",
        f"- $ per query: gated {_f(b['usd_per_query']['gated'], '.5f')}, S3 {_f(b['usd_per_query']['s3'], '.5f')}, "
        f"S1 {_f(b['usd_per_query']['s1'], '.5f')}; s per query gated {_f(b['s_per_query']['gated'], '.1f')}, S3 "
        f"{_f(b['s_per_query']['s3'], '.1f')}, S1 {_f(b['s_per_query']['s1'], '.1f')}",
        f"- outcome: **{b['outcome']}**",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args(argv)

    summary, rows = analyze(args.data_dir)
    write(args.results, summary, rows)
    print(report(summary))
    return summary


def write(results: Path, summary: dict, rows: list[dict]) -> None:
    (results / "private").mkdir(parents=True, exist_ok=True)
    (results / "summary.json").write_text(json.dumps(_clean(summary), indent=2) + "\n")
    (results / "analysis.md").write_text(report(summary))
    with (results / "private" / "per_query.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _clean(value):
    """Infinite thresholds (rate 1.0 = always search) as strings: standard JSON has no Infinity."""
    if isinstance(value, float) and math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


if __name__ == "__main__":
    main()
