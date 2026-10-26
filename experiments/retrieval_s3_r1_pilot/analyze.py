"""R1 pilot, free part: pair each R1 episode with the S3 pilot episode and the
reused S1 record for the same tune query, as the R1 proposal pre-registers.

    venv/bin/python experiments/retrieval_s3_r1_pilot/analyze.py

Reads, read-only: this study's episodes (main phase only), the S3 pilot's
episodes and public manifest (pinned SHA-256 in paid mode) and the reused S1
tune records, verified with the S3 pilot analyzer's pins (that module is
reused by import, pinned too). nDCG@10 per query for S0, S1, S3 (S3 + forced
rerank, from the pilot) and R1.

- primary, two-sided: R1 - S3, stratified with tune weights; readings:
  "S3's decision adds value" (CI upper < 0), "R1 better" (CI lower > 0),
  "equivalent within +-0.02" (CI inside [-0.02, +0.02]), else inconclusive;
- secondary: R1 - S1 overall and per stratum (within-20 margin -0.02),
  rescue and harm rates for R1 and S3, overlap of the documents their
  searches added, cost and latency per query;
- validity rules as in the S3 pilot (harness error, smoke, completeness,
  non-decline model errors above 5%, decline warning above 10%).

Writes results/summary.json and results/analysis.md (aggregates only) and
results/private/per_query.jsonl (label-derived, gitignored).
"""

import argparse
import importlib.util
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from oarp.envs.retrieval import study  # noqa: E402
from oarp.envs.retrieval.bm25 import BM25Config, BM25Index  # noqa: E402
from oarp.envs.retrieval.data import DEFAULT_DATA_DIR, dev_partition, load_scifact  # noqa: E402
from oarp.envs.retrieval.metrics import ndcg_at_k, relevant  # noqa: E402
from oarp.envs.retrieval.stats import paired_bootstrap_ci, stratified_bootstrap_ci  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PILOT_DIR = REPO / "experiments" / "retrieval_s3_pilot"
PILOT_ANALYZER_SHA256 = "b6a92b6bfd1ab49f1e40ca965579bc73e1a5a59a61e166edcc64320bf54d4334"
S3_EPISODES_SHA256 = "4f41bc759b8aad91752b696de57e4f8e52d92a1786692c01d289997ac8c94fbf"
BOOTSTRAP_SEED = 20261027
DECLINE = "ModelRefused"
BAND = 0.02  # equivalence band for R1 - S3, and the harm margin for R1 - S1 within 20
MAX_OTHER_ERROR_RATE = 0.05
DECLINE_WARNING_RATE = 0.10
DEPTH = 20


def load_pilot_analyzer():
    spec = importlib.util.spec_from_file_location("s3_pilot_analyzer_for_r1", PILOT_DIR / "analyze.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s3a = load_pilot_analyzer()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _mean(values) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def read_primary(ci: dict | None) -> dict | None:
    if ci is None:
        return None
    flags = {"s3_decision_adds_value": ci["high"] < 0, "r1_better": ci["low"] > 0,
             "equivalent_within_band": ci["low"] > -BAND and ci["high"] < BAND}
    reading = ("S3's decision adds value" if flags["s3_decision_adds_value"] else
               "R1 better" if flags["r1_better"] else
               f"equivalent within ±{BAND}" if flags["equivalent_within_band"] else "inconclusive")
    return {"reading": reading, **flags}


def classify(*, paid: bool, manifest: dict, other_error_rate: float, decline_rate: float, tool_bugs: int,
             primary: dict | None, r1_within: dict | None) -> dict[str, object]:
    if not paid:
        run = "dry run: no outcome"
    elif manifest["harness_error"] or tool_bugs:
        run = "invalid (harness error)"
    elif not manifest["smoke_passed"]:
        run = "stopped after the smoke phase (no result claimed)"
    elif manifest["stopped"] or manifest["completed_episodes"]["main"] < manifest["scheduled_episodes"]["main"]:
        run = "incomplete (no hypothesis claimed)"
    elif other_error_rate > MAX_OTHER_ERROR_RATE:
        run = "invalid (non-decline model errors above 5%)"
    else:
        run = "valid"
    valid = run == "valid"
    return {"run": run, "decline_warning": decline_rate > DECLINE_WARNING_RATE,
            "primary": read_primary(primary)["reading"] if valid and primary else None,
            "r1_within_no_harm_beyond_margin": (r1_within is not None and r1_within["low"] > -BAND) if valid else None}


def analyze(results: Path, pilot_results: Path, tau_results: Path, s3_results: Path,
            data_dir: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((results / "manifest.json").read_text())
    paid = manifest["mode"] == "paid"
    reused = {"pilot": s3a.verify_reused("pilot", pilot_results, paid=paid),
              "tau": s3a.verify_reused("tau", tau_results, paid=paid)}
    s3_path = s3_results / "private" / "episodes.jsonl"
    reused["s3_episodes"] = study.sha256_file(s3_path)
    reused["s3_pilot_analyzer"] = study.sha256_file(PILOT_DIR / "analyze.py")
    if paid and (reused["s3_episodes"] != S3_EPISODES_SHA256 or reused["s3_pilot_analyzer"] != PILOT_ANALYZER_SHA256):
        raise SystemExit("refusing: the S3 pilot episodes or analyzer differ from their pinned SHA-256")
    s1_rows = [r for r in _rows(pilot_results / "private" / "calls.jsonl") if r["pass"] == 1 and r["condition"] == "plain"]
    s1 = {r["query_id"]: r for r in s1_rows + _rows(tau_results / "private" / "calls.jsonl")}
    s3 = {r["query_id"]: r for r in _rows(s3_path) if r["phase"] == "main"}
    weights_src = json.loads((s3_results / "manifest.json").read_text())["strata_tune"]
    weights = {"beyond": weights_src["beyond"], "within": weights_src["within_available"]}
    episodes = [r for r in _rows(results / "private" / "episodes.jsonl") if r["phase"] == "main"]
    split = load_scifact(data_dir, "train")
    qrels = split.qrels
    tune = set(dev_partition(sorted(split.queries))["tune"])
    if any(r["query_id"] not in tune for r in episodes):
        raise SystemExit("refusing: an episode is not a tune query")
    unpaired = [r["query_id"] for r in episodes if r["query_id"] not in s3 or r["query_id"] not in s1]
    if unpaired:
        raise SystemExit(f"refusing: {len(unpaired)} R1 episodes have no S3 pilot episode or S1 record")

    def added(e: dict) -> list[str]:
        by_handle = {f"D{i}": d for i, d in enumerate(e["candidates"], 1)}
        return [by_handle[h] for s in e["searches"] if s["source"] == "model" for h in s["new_handles"]]

    def first_model_search(e: dict) -> set[str]:
        by_handle = {f"D{i}": d for i, d in enumerate(e["candidates"], 1)}
        model_searches = [s for s in e["searches"] if s["source"] == "model"]
        return {by_handle[h] for h in model_searches[0]["new_handles"]} if model_searches else set()

    index = BM25Index(split.corpus, BM25Config())
    rows = []
    for e in episodes:
        q, judged = e["query_id"], qrels[e["query_id"]]
        bm25 = [d for d, _ in index.rank(split.queries[q])]
        rec, s3e = s1[q], s3[q]
        if rec["bm25_top"] != bm25[:DEPTH] or e["candidates"][:DEPTH] != bm25[:DEPTH] \
                or s3e["candidates"][:DEPTH] != bm25[:DEPTH]:
            raise SystemExit("refusing: a recorded BM25 top 20 differs from a fresh BM25 ranking")
        if e["stratum"] != s3e["stratum"]:
            raise SystemExit("refusing: an R1 episode's stratum differs from its S3 pilot episode's")
        rel = relevant(judged)
        s1_rank = rec["ranking_top"] + bm25[DEPTH:]
        r1_added, s3_added = added(e), added(s3e)
        r1_first, s3_first = set(r1_added), first_model_search(s3e)
        union = r1_first | s3_first
        rr = e["rerank"] or {}
        declined = (e["error"] or "").startswith(DECLINE) or rr.get("error_kind") == DECLINE
        other_error = (e["stop_reason"] == "model_error" and not (e["error"] or "").startswith(DECLINE)) or (
            rr.get("status") == "model_error" and rr.get("error_kind") != DECLINE)
        r1_usd = e["call_usd"]["agent"] + e["call_usd"]["rerank"]
        s3_usd = s3e["call_usd"]["agent"] + s3e["call_usd"]["rerank"]
        rows.append({
            "query_id": q, "stratum": e["stratum"],
            "s0": ndcg_at_k(bm25, judged, 10), "s1": ndcg_at_k(s1_rank, judged, 10),
            "s3": ndcg_at_k(s3e["ranking"], judged, 10), "r1": ndcg_at_k(e["ranking"], judged, 10),
            "r1_rescued": any(d in rel for d in r1_added), "s3_rescued": any(d in rel for d in s3_added),
            "r1_lost": any(d in rel and d not in e["ranking"][:10] for d in s1_rank[:10]),
            "s3_lost": any(d in rel and d not in s3e["ranking"][:10] for d in s1_rank[:10]),
            "r1_added": len(r1_added), "s3_added": len(s3_added),
            "s3_searched": any(x["source"] == "model" for x in s3e["searches"]),
            "overlap_jaccard": len(r1_first & s3_first) / len(union) if union else None,
            "reformulation_status": e.get("reformulation_status"), "declined": declined, "other_error": other_error,
            "rerank_status": rr.get("status"), "rejected": e["tool_calls"]["rejected"],
            "r1_usd": sum(c for c in r1_usd if c is not None), "s3_usd": sum(c for c in s3_usd if c is not None),
            "unpriced_calls": sum(c is None for c in r1_usd),
            "s1_usd": rec.get("estimated_usd") or 0.0, "r1_wall_s": e["wall_s"], "s3_wall_s": s3e["wall_s"],
            "s1_latency_s": rec.get("latency_s") or 0.0,
        })

    strata = {"beyond": [r for r in rows if r["stratum"] == "beyond"],
              "within": [r for r in rows if r["stratum"] == "within"]}

    def stratified(a: str, b: str, keep=lambda r: True) -> dict | None:
        parts = {s: [r[a] - r[b] for r in rs if keep(r)] for s, rs in strata.items()}
        return None if any(not v for v in parts.values()) else stratified_bootstrap_ci(parts, weights, seed=BOOTSTRAP_SEED)

    def paired(rs: list[dict], a: str, b: str) -> dict | None:
        return paired_bootstrap_ci([r[a] - r[b] for r in rs], seed=BOOTSTRAP_SEED) if rs else None

    def weighted(key) -> float | None:
        means = {s: _mean(key(r) for r in rs) for s, rs in strata.items()}
        return None if any(v is None for v in means.values()) else \
            sum(weights[s] * means[s] for s in strata) / sum(weights.values())

    primary = stratified("r1", "s3")
    r1_s1 = {"overall": stratified("r1", "s1"), "beyond": paired(strata["beyond"], "r1", "s1"),
             "within": paired(strata["within"], "r1", "s1")}
    r1_s3_strata = {"beyond": paired(strata["beyond"], "r1", "s3"), "within": paired(strata["within"], "r1", "s3")}
    rejected: Counter = Counter()
    for r in rows:
        rejected.update(r["rejected"])
    overlaps = [r["overlap_jaccard"] for r in rows if r["overlap_jaccard"] is not None]
    n = len(rows)
    other_rate = sum(r["other_error"] for r in rows) / n if n else 0.0
    decline_rate = sum(r["declined"] for r in rows) / n if n else 0.0
    summary = {
        "study": manifest["study"], "mode": manifest["mode"], "git_at_run": manifest["git"],
        "episodes": {"scheduled_main": manifest["scheduled_episodes"]["main"], "scored": n,
                     "per_stratum": {s: len(rs) for s, rs in strata.items()}},
        "smoke_passed": manifest["smoke_passed"], "stopped": manifest["stopped"],
        "stop_reason": manifest["stop_reason"], "harness_error": manifest["harness_error"],
        "estimated_spend_usd": manifest["estimated_spend_usd"], "reused_sha256": reused,
        "weights_tune": weights, "bootstrap_seed": BOOTSTRAP_SEED, "band": BAND,
        "means": {s: {c: _mean(r[c] for r in rs) for c in ("s0", "s1", "s3", "r1")} for s, rs in strata.items()},
        "primary_r1_minus_s3": primary, "primary_reading": read_primary(primary),
        "r1_minus_s3_by_stratum": r1_s3_strata, "r1_minus_s1": r1_s1,
        "mechanism": {
            "rescue_rate_beyond": {"r1": _mean(r["r1_rescued"] for r in strata["beyond"]),
                                   "s3": _mean(r["s3_rescued"] for r in strata["beyond"])},
            "harm_rate_within": {"r1": _mean(r["r1_lost"] for r in strata["within"]),
                                 "s3": _mean(r["s3_lost"] for r in strata["within"])},
            "mean_new_documents": {s: {"r1": _mean(r["r1_added"] for r in rs), "s3": _mean(r["s3_added"] for r in rs)}
                                   for s, rs in strata.items()},
            "s3_searched_share": {s: _mean(r["s3_searched"] for r in rs) for s, rs in strata.items()},
            "overlap_jaccard_first_search": {"mean": _mean(overlaps), "episodes": len(overlaps),
                                             "note": "R1's new documents vs those of S3's first model search"},
        },
        "behaviour": {"reformulation_status": dict(Counter(r["reformulation_status"] for r in rows)),
                      "rerank_status": dict(Counter(r["rerank_status"] for r in rows)),
                      "rejected_tool_calls": dict(rejected)},
        "cost_latency": {
            "usd_per_query_weighted": {"r1": weighted(lambda r: r["r1_usd"]), "s3": weighted(lambda r: r["s3_usd"]),
                                       "s1": weighted(lambda r: r["s1_usd"])},
            "s_per_query_weighted": {"r1": weighted(lambda r: r["r1_wall_s"]), "s3": weighted(lambda r: r["s3_wall_s"]),
                                     "s1": weighted(lambda r: r["s1_latency_s"])},
            "unpriced_calls": sum(r["unpriced_calls"] for r in rows),
            "note": "estimated from usage with pricing.py; unpriced calls (errors) count as $0 here"},
        "sensitivity": {"primary_excluding_declined": stratified("r1", "s3", keep=lambda r: not r["declined"])},
        "errors": {"declined_episodes": sum(r["declined"] for r in rows), "decline_rate": decline_rate,
                   "other_error_episodes": sum(r["other_error"] for r in rows), "other_error_rate": other_rate},
    }
    summary["outcome_rules"] = classify(paid=paid, manifest=manifest, other_error_rate=other_rate,
                                        decline_rate=decline_rate, tool_bugs=rejected.get("tool_bug", 0),
                                        primary=primary, r1_within=r1_s1["within"])
    return summary, rows


def _f(value, spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value, spec)


def report(s: dict) -> str:
    def ci(c: dict | None) -> str:
        return "n/a" if c is None else f"{c['mean']:+.3f} [{c['low']:+.3f}, {c['high']:+.3f}]"

    m, c, e = s["mechanism"], s["cost_latency"], s["episodes"]
    lines = [
        f"# R1 pilot: reformulate once, then rerank ({s['mode']})", "",
        f"Episodes scored: {e['scored']}/{e['scheduled_main']} ({e['per_stratum']}); smoke passed: "
        f"{s['smoke_passed']}; stopped: {s['stopped']} ({s['stop_reason']}); estimated spend "
        f"${s['estimated_spend_usd']:.4f}. Outcome rules: {s['outcome_rules']}.", "",
        "Tune half (selection data), one episode per query, paired with the S3 pilot's episodes and the reused "
        "S1 records for the same queries.", "",
        "## nDCG@10 means", "", "| stratum | S0 | S1 | S3 (pilot) | R1 |", "|---|---|---|---|---|",
        *[f"| {st} | {_f(v['s0'])} | {_f(v['s1'])} | {_f(v['s3'])} | {_f(v['r1'])} |" for st, v in s["means"].items()],
        "", "## Comparisons (95% CI)", "",
        f"- **primary** R1 − S3 (stratified, tune weights {s['weights_tune']}): {ci(s['primary_r1_minus_s3'])} "
        f"→ {s['primary_reading']}",
        f"- R1 − S3 by stratum: beyond {ci(s['r1_minus_s3_by_stratum']['beyond'])}, within "
        f"{ci(s['r1_minus_s3_by_stratum']['within'])}",
        f"- R1 − S1: overall {ci(s['r1_minus_s1']['overall'])}, beyond {ci(s['r1_minus_s1']['beyond'])}, within "
        f"{ci(s['r1_minus_s1']['within'])} (harm margin −{s['band']})", "",
        "## Mechanism (labels, analysis only)", "",
        f"- rescue rate (beyond): R1 {_f(m['rescue_rate_beyond']['r1'])}, S3 {_f(m['rescue_rate_beyond']['s3'])}",
        f"- harm (within: a relevant document in S1's top 10 lost): R1 {_f(m['harm_rate_within']['r1'])}, S3 "
        f"{_f(m['harm_rate_within']['s3'])}",
        f"- new documents per episode: {m['mean_new_documents']}; S3 searched at all: {m['s3_searched_share']}",
        f"- overlap of added documents (Jaccard, R1 vs S3's first search): "
        f"{_f(m['overlap_jaccard_first_search']['mean'])} over {m['overlap_jaccard_first_search']['episodes']} episodes",
        f"- behaviour: {s['behaviour']}", "",
        "## Cost and latency (estimates, tune-weighted per query)", "",
        f"- $: R1 {_f(c['usd_per_query_weighted']['r1'], '.5f')}, S3 {_f(c['usd_per_query_weighted']['s3'], '.5f')}, "
        f"S1 {_f(c['usd_per_query_weighted']['s1'], '.5f')}; unpriced calls {c['unpriced_calls']}",
        f"- s: R1 {_f(c['s_per_query_weighted']['r1'], '.1f')}, S3 {_f(c['s_per_query_weighted']['s3'], '.1f')}, "
        f"S1 {_f(c['s_per_query_weighted']['s1'], '.1f')}", "",
        "## Sensitivity", "",
        f"- primary excluding declined episodes: {ci(s['sensitivity']['primary_excluding_declined'])}",
        f"- errors: {s['errors']}",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--pilot-results", type=Path, default=s3a.PILOT_RESULTS)
    parser.add_argument("--tau-results", type=Path, default=s3a.TAU_RESULTS)
    parser.add_argument("--s3-results", type=Path, default=PILOT_DIR / "results")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)

    summary, rows = analyze(args.results, args.pilot_results, args.tau_results, args.s3_results, args.data_dir)
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
