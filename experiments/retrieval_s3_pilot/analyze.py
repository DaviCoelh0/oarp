"""S3 pilot, free part: score the episodes against the reused S1 records, as the
approved contract pre-registers.

    venv/bin/python experiments/retrieval_s3_pilot/analyze.py

Reads, read-only: this study's episodes (main phase only; smoke episodes are
excluded) and the reused S1-plain records for the tune half (S1 pilot pass 1:
100 queries; S2 threshold study: 305), which in paid mode must match their
pinned SHA-256. Conditions per query: S0 (BM25), S1 (the record's reranked top
20, then BM25's remainder), S3 + forced rerank (the episode's final ranking)
and S3 + agent ordering (the agent's RANKING over the same candidates, then
BM25's remainder), all scored with nDCG@10.

- H1: stratified S3+FR - S1 (weights = tune shares of the strata);
- H2: beyond stratum S3+FR - S1 > 0; H3: within stratum S3+FR - S1 > -0.02;
- H4: S3+AO - S3+FR, overall (stratified) and per stratum, two-sided;
- the next-step rule and the section-8 outcome rules, applied mechanically;
- model behaviour, mechanism (rescue, landing, harm; RM3 as context), cost
  and latency, and the sensitivity analyses.

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

from oarp.envs.retrieval import study  # noqa: E402
from oarp.envs.retrieval.bm25 import BM25Config, BM25Index  # noqa: E402
from oarp.envs.retrieval.data import DEFAULT_DATA_DIR, dev_partition, load_scifact  # noqa: E402
from oarp.envs.retrieval.metrics import first_relevant_rank, ndcg_at_k, relevant  # noqa: E402
from oarp.envs.retrieval.rm3 import RM3Config, rm3_rank  # noqa: E402
from oarp.envs.retrieval.stats import paired_bootstrap_ci, stratified_bootstrap_ci  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PILOT_RESULTS = REPO / "experiments" / "retrieval_s1_pilot" / "results"
TAU_RESULTS = REPO / "experiments" / "retrieval_s2_tau" / "results"
REUSED_FILES = ("manifest.json", "private/sample.json", "private/calls.jsonl")
PINNED = {
    "pilot": {"manifest.json": "a8013b6d368c50ad98a96b2ba22d792431e1dc3ec19e78658329e23e7ec45cec",
              "private/sample.json": "771a2020c58ef37aa59b258ca351eb7c7a0f382b71676c8be86f8e9a12b48021",
              "private/calls.jsonl": "4a143981f208ea8fefc5fa31ad53d6fa370f16c468518b7c63bdca1714acc50e"},
    "tau": {"manifest.json": "864868fca71bdf06f880f4e30b4c600f5742b11e869f5b609165178cdebbe8d9",
            "private/sample.json": "4cb7624657a8f0d07c29e6cb68577544ed84645e30f81fdade10080bd1e40ef0",
            "private/calls.jsonl": "7395477e2c05603c39b14f170445977015589d9d3b0d48060e5f4859dd37a0c0"},
}
BOOTSTRAP_SEED = 20261025
DECLINE = "ModelRefused"
MARGIN = -0.02
MAX_OTHER_ERROR_RATE = 0.05
DECLINE_WARNING_RATE = 0.10
CHECK_WEIGHTS = {"beyond": 55, "within": 349}  # check-half strata, for the sensitivity analysis
DEPTH = 20  # the forced first search (AgentConfig.first_k)


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


def read_h(ci: dict | None) -> str | None:
    """Two-sided reading of a difference CI."""
    if ci is None:
        return None
    return "positive" if ci["low"] > 0 else "negative" if ci["high"] < 0 else "inconclusive"


def classify(*, paid: bool, manifest: dict, other_error_rate: float, decline_rate: float, tool_bugs: int,
             h1: dict | None, h2: dict | None, h3: dict | None) -> dict[str, object]:
    """Section 8 of the contract and the next-step rule of section 2, applied mechanically."""
    scheduled = manifest["scheduled_episodes"]["main"]
    if not paid:
        run = "dry run: no outcome"
    elif manifest["harness_error"] or tool_bugs:
        run = "invalid (harness error)"
    elif not manifest["smoke_passed"]:
        run = "stopped after the smoke phase (no result claimed)"
    elif manifest["stopped"] or manifest["completed_episodes"]["main"] < scheduled:
        run = "incomplete (no hypothesis claimed)"
    elif other_error_rate > MAX_OTHER_ERROR_RATE:
        run = "invalid (non-decline model errors above 5%)"
    else:
        run = "valid"
    valid = run == "valid"
    rules = {"run": run, "decline_warning": decline_rate > DECLINE_WARNING_RATE,
             "H1": read_h(h1) if valid else None,
             "H2_holds": (h2 is not None and h2["low"] > 0) if valid else None,
             "H3_holds": (h3 is not None and h3["low"] > MARGIN) if valid else None}
    rules["earns_confirmation"] = (rules["H1"] == "positive" or (
        bool(rules["H2_holds"]) and bool(rules["H3_holds"]) and h1 is not None and h1["mean"] > 0)) if valid else None
    return rules


def analyze(results: Path, pilot_results: Path, tau_results: Path, data_dir: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((results / "manifest.json").read_text())
    paid = manifest["mode"] == "paid"
    reused = {"pilot": verify_reused("pilot", pilot_results, paid=paid),
              "tau": verify_reused("tau", tau_results, paid=paid)}
    s1_rows = [r for r in _rows(pilot_results / "private" / "calls.jsonl") if r["pass"] == 1 and r["condition"] == "plain"]
    s1_rows += _rows(tau_results / "private" / "calls.jsonl")
    s1 = {r["query_id"]: r for r in s1_rows}
    episodes = [r for r in _rows(results / "private" / "episodes.jsonl") if r["phase"] == "main"]
    split = load_scifact(data_dir, "train")
    qrels = split.qrels
    tune = set(dev_partition(sorted(split.queries))["tune"])
    if any(r["query_id"] not in tune for r in episodes):
        raise SystemExit("refusing: an episode is not a tune query")
    missing = [r["query_id"] for r in episodes if r["query_id"] not in s1]
    if missing:
        raise SystemExit(f"refusing: {len(missing)} episodes have no reused S1 record")

    index = BM25Index(split.corpus, BM25Config())
    rows = []
    for e in episodes:
        q, judged = e["query_id"], qrels[e["query_id"]]
        bm25 = [d for d, _ in index.rank(split.queries[q])]
        rec = s1[q]
        if rec["bm25_top"] != bm25[:DEPTH] or e["candidates"][:DEPTH] != bm25[:DEPTH]:
            raise SystemExit("refusing: a recorded BM25 top 20 differs from a fresh BM25 ranking")
        s1_rank = rec["ranking_top"] + bm25[DEPTH:]
        candidates = set(e["candidates"])
        agent = e["agent_ranking"] + [d for d in bm25 if d not in candidates]
        final = e["ranking"]
        rel = relevant(judged)
        by_handle = {f"D{i}": d for i, d in enumerate(e["candidates"], 1)}
        added = [by_handle[h] for s in e["searches"] if s["source"] == "model" for h in s["new_handles"]]
        rescued = [d for d in added if d in rel]
        rr = e["rerank"] or {}
        declined = (e["error"] or "").startswith(DECLINE) or rr.get("error_kind") == DECLINE
        other_error = (e["stop_reason"] == "model_error" and not (e["error"] or "").startswith(DECLINE)) or (
            rr.get("status") == "model_error" and rr.get("error_kind") != DECLINE)
        agent_usd = [c for c in e["call_usd"]["agent"]]
        rerank_usd = [c for c in e["call_usd"]["rerank"]]
        rows.append({
            "query_id": q, "stratum": e["stratum"],
            "s0": ndcg_at_k(bm25, judged, 10), "s1": ndcg_at_k(s1_rank, judged, 10),
            "s3fr": ndcg_at_k(final, judged, 10), "s3ao": ndcg_at_k(agent, judged, 10),
            "searches": e["tool_calls"]["executed_searches"], "added": len(added), "rescued": len(rescued) > 0,
            "landed_fr": any(d in final[:10] for d in rescued), "landed_ao": any(d in agent[:10] for d in rescued),
            "lost_fr": any(d in rel and d not in final[:10] for d in s1_rank[:10]),
            "lost_ao": any(d in rel and d not in agent[:10] for d in s1_rank[:10]),
            "declined": declined, "other_error": other_error, "stop_reason": e["stop_reason"],
            "agent_status": e["agent_ranking_status"], "rerank_status": rr.get("status"),
            "rejected": e["tool_calls"]["rejected"],
            "agent_usd": sum(c for c in agent_usd if c is not None),
            "rerank_usd": sum(c for c in rerank_usd if c is not None),
            "unpriced_calls": sum(c is None for c in agent_usd + rerank_usd),
            "s1_usd": rec.get("estimated_usd"), "s1_latency_s": rec.get("latency_s"), "wall_s": e["wall_s"],
            "model_calls": e["model_calls"],
        })

    strata = {"beyond": [r for r in rows if r["stratum"] == "beyond"],
              "within": [r for r in rows if r["stratum"] == "within"]}
    tune_counts = manifest["strata_tune"]
    weights = {"beyond": tune_counts["beyond"], "within": tune_counts["within_available"]}

    def diffs(rs: list[dict], a: str, b: str) -> list[float]:
        return [r[a] - r[b] for r in rs]

    def paired(rs: list[dict], a: str, b: str) -> dict | None:
        return paired_bootstrap_ci(diffs(rs, a, b), seed=BOOTSTRAP_SEED) if rs else None

    def stratified(a: str, b: str, w: dict = weights, keep=lambda r: True) -> dict | None:
        parts = {s: [r[a] - r[b] for r in rs if keep(r)] for s, rs in strata.items()}
        if any(not v for v in parts.values()):
            return None
        return stratified_bootstrap_ci(parts, w, seed=BOOTSTRAP_SEED)

    h1 = stratified("s3fr", "s1")
    h2 = paired(strata["beyond"], "s3fr", "s1")
    h3 = paired(strata["within"], "s3fr", "s1")
    h4 = {"overall": stratified("s3ao", "s3fr"), "beyond": paired(strata["beyond"], "s3ao", "s3fr"),
          "within": paired(strata["within"], "s3ao", "s3fr")}
    h4_reading = {k: read_h(v) for k, v in h4.items()}

    def share(rs: list[dict], key: str) -> float | None:
        return _mean(r[key] for r in rs)

    beyond_ids = [r["query_id"] for r in strata["beyond"]]
    rm3 = {q: first_relevant_rank([d for d, _ in rm3_rank(index, split.corpus, split.queries[q], RM3Config())],
                                  qrels[q]) for q in beyond_ids}
    rescued = [r for r in strata["beyond"] if r["rescued"]]
    mechanism = {
        "rescue_rate_beyond": share(strata["beyond"], "rescued"), "rescued": len(rescued),
        "landing_rate_forced_rerank": share(rescued, "landed_fr"), "landing_rate_agent": share(rescued, "landed_ao"),
        "harm_rate_within_forced_rerank": share(strata["within"], "lost_fr"),
        "harm_rate_within_agent": share(strata["within"], "lost_ao"),
        "rm3_context_beyond": {"first_relevant_within_10": _mean(r is not None and r <= 10 for r in rm3.values()),
                               "first_relevant_within_20": _mean(r is not None and r <= 20 for r in rm3.values()),
                               "note": "descriptive: RM3 is one fixed ranking; the agent adds up to 2 x 10 results"},
    }
    rejected: Counter = Counter()
    for r in rows:
        rejected.update(r["rejected"])
    behaviour = {s: {"episodes": len(rs), "mean_searches": _mean(r["searches"] for r in rs),
                     "searches": dict(sorted(Counter(r["searches"] for r in rs).items())),
                     "share_no_search": _mean(r["searches"] == 0 for r in rs),
                     "mean_new_documents": _mean(r["added"] for r in rs),
                     "mean_model_calls": _mean(r["model_calls"] for r in rs)} for s, rs in strata.items()}
    behaviour["stop_reasons"] = dict(Counter(r["stop_reason"] for r in rows))
    behaviour["agent_ranking_status"] = dict(Counter(r["agent_status"] for r in rows))
    behaviour["rerank_status"] = dict(Counter(r["rerank_status"] for r in rows))
    behaviour["rejected_tool_calls"] = dict(rejected)

    def total(r: dict) -> float:
        return r["agent_usd"] + r["rerank_usd"]

    def weighted_mean(key) -> float | None:
        means = {s: _mean(key(r) for r in rs) for s, rs in strata.items()}
        if any(v is None for v in means.values()):
            return None
        return sum(weights[s] * means[s] for s in strata) / sum(weights.values())

    s3_cost = weighted_mean(total)
    s1_cost = weighted_mean(lambda r: r["s1_usd"] or 0.0)
    population_rescue = (weights["beyond"] / sum(weights.values())) * (mechanism["rescue_rate_beyond"] or 0.0)
    cost = {
        "per_stratum_usd": {s: _mean(total(r) for r in rs) for s, rs in strata.items()},
        "s3_per_query_usd_weighted": s3_cost, "s1_per_query_usd_weighted": s1_cost,
        "extra_usd_per_rescued_query": ((s3_cost - s1_cost) / population_rescue)
        if s3_cost is not None and s1_cost is not None and population_rescue > 0 else None,
        "rerank_share_of_cost": (sum(r["rerank_usd"] for r in rows) / sum(total(r) for r in rows))
        if rows and sum(total(r) for r in rows) > 0 else None,
        "unpriced_calls": sum(r["unpriced_calls"] for r in rows),
        "s3_wall_s_per_query_weighted": weighted_mean(lambda r: r["wall_s"]),
        "s1_latency_s_per_query_weighted": weighted_mean(lambda r: r["s1_latency_s"] or 0.0),
        "per_stratum_wall_s": {s: _mean(r["wall_s"] for r in rs) for s, rs in strata.items()},
        "note": "estimated from usage with pricing.py; unpriced calls (errors) count as $0 here",
    }
    sensitivity = {
        "h1_excluding_declined": stratified("s3fr", "s1", keep=lambda r: not r["declined"]),
        "within_unweighted": h3,
        "h1_with_check_weights": stratified("s3fr", "s1", w=CHECK_WEIGHTS),
    }
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
        "weights_tune": weights, "bootstrap_seed": BOOTSTRAP_SEED, "margin_h3": MARGIN,
        "means": {s: {c: _mean(r[c] for r in rs) for c in ("s0", "s1", "s3fr", "s3ao")} for s, rs in strata.items()},
        "H1_overall_s3fr_minus_s1": h1, "H2_beyond_s3fr_minus_s1": h2, "H3_within_s3fr_minus_s1": h3,
        "H4_s3ao_minus_s3fr": h4, "H4_reading": h4_reading,
        "mechanism": mechanism, "behaviour": behaviour, "cost_latency": cost, "sensitivity": sensitivity,
        "errors": {"declined_episodes": sum(r["declined"] for r in rows), "decline_rate": decline_rate,
                   "other_error_episodes": sum(r["other_error"] for r in rows), "other_error_rate": other_rate},
    }
    summary["outcome_rules"] = classify(paid=paid, manifest=manifest, other_error_rate=other_rate,
                                        decline_rate=decline_rate, tool_bugs=rejected.get("tool_bug", 0),
                                        h1=h1, h2=h2, h3=h3)
    return summary, rows


def _f(value, spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value, spec)


def report(s: dict) -> str:
    def ci(c: dict | None) -> str:
        return "n/a" if c is None else f"{c['mean']:+.3f} [{c['low']:+.3f}, {c['high']:+.3f}]"

    m, b, c, e = s["mechanism"], s["behaviour"], s["cost_latency"], s["episodes"]
    lines = [
        f"# S3 pilot ({s['mode']})", "",
        f"Episodes scored: {e['scored']}/{e['scheduled_main']} ({e['per_stratum']}); smoke passed: "
        f"{s['smoke_passed']}; stopped: {s['stopped']} ({s['stop_reason']}); estimated spend "
        f"${s['estimated_spend_usd']:.4f}. Outcome rules: {s['outcome_rules']}.", "",
        "Tune half (selection data), one episode per query; S1 reused from earlier runs. Not a confirmation.", "",
        "## nDCG@10 means", "", "| stratum | S0 | S1 | S3 + forced rerank | S3 + agent ordering |", "|---|---|---|---|---|",
        *[f"| {st} | {_f(v['s0'])} | {_f(v['s1'])} | {_f(v['s3fr'])} | {_f(v['s3ao'])} |" for st, v in s["means"].items()],
        "", "## Hypotheses (95% CI)", "",
        f"- H1 overall (stratified, tune weights {s['weights_tune']}), S3+FR − S1: {ci(s['H1_overall_s3fr_minus_s1'])}",
        f"- H2 beyond, S3+FR − S1: {ci(s['H2_beyond_s3fr_minus_s1'])}",
        f"- H3 within, S3+FR − S1 (margin {s['margin_h3']}): {ci(s['H3_within_s3fr_minus_s1'])}",
        f"- H4 S3+AO − S3+FR: overall {ci(s['H4_s3ao_minus_s3fr']['overall'])}, beyond "
        f"{ci(s['H4_s3ao_minus_s3fr']['beyond'])}, within {ci(s['H4_s3ao_minus_s3fr']['within'])} "
        f"→ {s['H4_reading']}", "",
        "## Mechanism (labels, analysis only)", "",
        f"- rescue rate (beyond: a model search added a relevant document): {_f(m['rescue_rate_beyond'])} "
        f"({m['rescued']} queries); landing in the top 10: forced rerank {_f(m['landing_rate_forced_rerank'])}, "
        f"agent {_f(m['landing_rate_agent'])}",
        f"- harm (within: a relevant document in S1's top 10 lost): forced rerank "
        f"{_f(m['harm_rate_within_forced_rerank'])}, agent {_f(m['harm_rate_within_agent'])}",
        f"- RM3 context (beyond, same queries): first relevant in its top 10 "
        f"{_f(m['rm3_context_beyond']['first_relevant_within_10'])}, top 20 "
        f"{_f(m['rm3_context_beyond']['first_relevant_within_20'])} ({m['rm3_context_beyond']['note']})", "",
        "## Model behaviour (reported separately from harness settings)", "",
        *[f"- {st}: {b[st]}" for st in ("beyond", "within")],
        f"- stop reasons {b['stop_reasons']}; agent ranking {b['agent_ranking_status']}; rerank "
        f"{b['rerank_status']}; rejected tool calls {b['rejected_tool_calls']}", "",
        "## Cost and latency (estimates)", "",
        f"- $ per query (tune-weighted): S3 {_f(c['s3_per_query_usd_weighted'], '.5f')} vs S1 "
        f"{_f(c['s1_per_query_usd_weighted'], '.5f')}; per stratum {c['per_stratum_usd']}",
        f"- extra $ per rescued query {_f(c['extra_usd_per_rescued_query'], '.4f')}; rerank share of cost "
        f"{_f(c['rerank_share_of_cost'])}; unpriced calls {c['unpriced_calls']}",
        f"- s per query (tune-weighted): S3 {_f(c['s3_wall_s_per_query_weighted'], '.1f')} vs S1 "
        f"{_f(c['s1_latency_s_per_query_weighted'], '.1f')}", "",
        "## Sensitivity", "",
        f"- H1 excluding declined episodes: {ci(s['sensitivity']['h1_excluding_declined'])}",
        f"- H1 with check-half weights {CHECK_WEIGHTS}: {ci(s['sensitivity']['h1_with_check_weights'])}",
        f"- errors: {s['errors']}",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--pilot-results", type=Path, default=PILOT_RESULTS)
    parser.add_argument("--tau-results", type=Path, default=TAU_RESULTS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)

    summary, rows = analyze(args.results, args.pilot_results, args.tau_results, args.data_dir)
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
