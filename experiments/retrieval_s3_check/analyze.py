"""S3 confirmation, free part: score the check-half episodes against the S1
check study's records, as the approved contract pre-registers.

    venv/bin/python experiments/retrieval_s3_check/analyze.py

Reads, read-only: this study's episodes (main phase only; the smoke episodes
on tune are excluded), the S1 check study's pass-1 records (all 404 check
queries; pinned SHA-256 in paid mode) and the S3 pilot's public summary (for
the descriptive tune -> check row). Conditions per query: S0 (BM25), S1 (the
record's reranked top 20, then BM25's remainder), S3 + forced rerank (the
episode's final ranking) and S3 + agent ordering (the agent's RANKING over the
same candidates, then BM25's remainder), scored with nDCG@10.

- C1: S3+FR - S1 over all check queries (paired bootstrap, seed 20261026);
- C2: within-20 stratum, lower bound > -0.02; C3: beyond-20 stratum, > 0;
- C4: S3+AO - S3+FR, descriptive;
- outcome: confirmed (C1 and C2) / confirmed with harm (C1 only) / not
  confirmed, after the validity rules of section 8.

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
from oarp.envs.retrieval.metrics import ndcg_at_k, relevant  # noqa: E402
from oarp.envs.retrieval.stats import paired_bootstrap_ci  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
S1_CHECK = REPO / "experiments" / "retrieval_s1_check" / "results"
S1_CHECK_SHA256 = {"manifest.json": "d0905eb86fbf3d973aa3bab69df0a3531327aa97f763ce01c9400539637c0220",
                   "private/sample.json": "e13ab3132ea07c933c62fdfc0ea718aa7fb26bcb369d8673b260f43501fe6238",
                   "private/calls.jsonl": "f0ff07cba7194c208848fbd6036772f13f41ded5da342fcfb068378cf064cb19"}
PILOT_SUMMARY = REPO / "experiments" / "retrieval_s3_pilot" / "results" / "summary.json"
BOOTSTRAP_SEED = 20261026
DECLINE = "ModelRefused"
MARGIN = -0.02
MAX_OTHER_ERROR_RATE = 0.05
DECLINE_WARNING_RATE = 0.10
DEPTH = 20


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _mean(values) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def verify_s1(folder: Path, *, paid: bool) -> dict[str, str]:
    missing = [f for f in S1_CHECK_SHA256 if not (folder / f).is_file()]
    if missing:
        raise SystemExit(f"refusing: S1 check files missing: {missing}")
    hashes = {f: study.sha256_file(folder / f) for f in S1_CHECK_SHA256}
    if paid:
        changed = [f for f in S1_CHECK_SHA256 if hashes[f] != S1_CHECK_SHA256[f]]
        if changed:
            raise SystemExit(f"refusing: S1 check files differ from the pinned SHA-256: {changed}")
    return hashes


def classify(*, paid: bool, manifest: dict, other_error_rate: float, decline_rate: float, tool_bugs: int,
             c1: dict | None, c2: dict | None) -> dict[str, object]:
    """Section 8 (validity) and the outcome rule of section 2, applied mechanically."""
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
    outcome = None
    if run == "valid":
        c1_holds = c1 is not None and c1["low"] > 0
        c2_holds = c2 is not None and c2["low"] > MARGIN
        outcome = "confirmed" if c1_holds and c2_holds else "confirmed with harm" if c1_holds else "not confirmed"
    return {"run": run, "outcome": outcome, "decline_warning": decline_rate > DECLINE_WARNING_RATE}


def read_h(ci: dict | None) -> str | None:
    if ci is None:
        return None
    return "positive" if ci["low"] > 0 else "negative" if ci["high"] < 0 else "inconclusive"


def analyze(results: Path, s1_check: Path, data_dir: Path, pilot_summary: Path = PILOT_SUMMARY) -> tuple[dict, list[dict]]:
    manifest = json.loads((results / "manifest.json").read_text())
    paid = manifest["mode"] == "paid"
    reused = verify_s1(s1_check, paid=paid)
    s1 = {r["query_id"]: r for r in _rows(s1_check / "private" / "calls.jsonl") if r["pass"] == 1}
    episodes = [r for r in _rows(results / "private" / "episodes.jsonl") if r["phase"] == "main"]
    split = load_scifact(data_dir, "train")
    qrels = split.qrels
    check = set(dev_partition(sorted(split.queries))["check"])
    if any(r["query_id"] not in check for r in episodes):
        raise SystemExit("refusing: a main episode is not a check query")
    missing = [r["query_id"] for r in episodes if r["query_id"] not in s1]
    if missing:
        raise SystemExit(f"refusing: {len(missing)} episodes have no S1 check record")

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
        usd = e["call_usd"]["agent"] + e["call_usd"]["rerank"]
        rows.append({
            "query_id": q, "stratum": e["stratum"],
            "s0": ndcg_at_k(bm25, judged, 10), "s1": ndcg_at_k(s1_rank, judged, 10),
            "s3fr": ndcg_at_k(final, judged, 10), "s3ao": ndcg_at_k(agent, judged, 10),
            "searches": e["tool_calls"]["executed_searches"], "added": len(added), "rescued": bool(rescued),
            "landed_fr": any(d in final[:10] for d in rescued), "landed_ao": any(d in agent[:10] for d in rescued),
            "lost_fr": any(d in rel and d not in final[:10] for d in s1_rank[:10]),
            "declined": declined, "other_error": other_error, "stop_reason": e["stop_reason"],
            "agent_status": e["agent_ranking_status"], "rerank_status": rr.get("status"),
            "rejected": e["tool_calls"]["rejected"], "model_calls": e["model_calls"],
            "usd": sum(c for c in usd if c is not None),
            "rerank_usd": sum(c for c in e["call_usd"]["rerank"] if c is not None),
            "unpriced_calls": sum(c is None for c in usd),
            "s1_usd": rec.get("estimated_usd"), "s1_latency_s": rec.get("latency_s"), "wall_s": e["wall_s"],
        })

    strata = {"beyond": [r for r in rows if r["stratum"] == "beyond"],
              "within": [r for r in rows if r["stratum"] == "within"]}

    def paired(rs: list[dict], a: str, b: str) -> dict | None:
        return paired_bootstrap_ci([r[a] - r[b] for r in rs], seed=BOOTSTRAP_SEED) if rs else None

    c1 = paired(rows, "s3fr", "s1")
    c2 = paired(strata["within"], "s3fr", "s1")
    c3 = paired(strata["beyond"], "s3fr", "s1")
    c4 = {"overall": paired(rows, "s3ao", "s3fr"), "beyond": paired(strata["beyond"], "s3ao", "s3fr"),
          "within": paired(strata["within"], "s3ao", "s3fr")}
    total_gain = sum(r["s3fr"] - r["s1"] for r in rows)
    gain_share = {s: (sum(r["s3fr"] - r["s1"] for r in rs) / total_gain) if total_gain else None
                  for s, rs in strata.items()}
    rescued = [r for r in strata["beyond"] if r["rescued"]]
    rejected: Counter = Counter()
    for r in rows:
        rejected.update(r["rejected"])
    behaviour = {s: {"episodes": len(rs), "mean_searches": _mean(r["searches"] for r in rs),
                     "searches": dict(sorted(Counter(r["searches"] for r in rs).items())),
                     "share_no_search": _mean(r["searches"] == 0 for r in rs),
                     "mean_new_documents": _mean(r["added"] for r in rs),
                     "mean_model_calls": _mean(r["model_calls"] for r in rs)} for s, rs in strata.items()}
    behaviour.update(stop_reasons=dict(Counter(r["stop_reason"] for r in rows)),
                     agent_ranking_status=dict(Counter(r["agent_status"] for r in rows)),
                     rerank_status=dict(Counter(r["rerank_status"] for r in rows)),
                     rejected_tool_calls=dict(rejected))
    s3_cost, s1_cost = _mean(r["usd"] for r in rows), _mean(r["s1_usd"] or 0.0 for r in rows)
    cost = {"s3_per_query_usd": s3_cost, "s1_per_query_usd": s1_cost,
            "per_stratum_usd": {s: _mean(r["usd"] for r in rs) for s, rs in strata.items()},
            "extra_usd_per_rescued_query": (len(rows) * (s3_cost - s1_cost) / len(rescued))
            if rows and rescued else None,
            "rerank_share_of_cost": (sum(r["rerank_usd"] for r in rows) / sum(r["usd"] for r in rows))
            if rows and sum(r["usd"] for r in rows) > 0 else None,
            "unpriced_calls": sum(r["unpriced_calls"] for r in rows),
            "s3_wall_s_per_query": _mean(r["wall_s"] for r in rows),
            "s1_latency_s_per_query": _mean(r["s1_latency_s"] or 0.0 for r in rows),
            "note": "estimated from usage with pricing.py; unpriced calls (errors) count as $0 here"}
    pilot = json.loads(pilot_summary.read_text()) if pilot_summary.is_file() else None
    tune_to_check = {
        "pilot_tune_reweighted_to_check": pilot["sensitivity"]["h1_with_check_weights"] if pilot else None,
        "check": c1, "pilot_summary_sha256": study.sha256_file(pilot_summary) if pilot else None,
        "note": "descriptive: the pilot is a stratified tune sample reweighted to check proportions"}
    n = len(rows)
    other_rate = sum(r["other_error"] for r in rows) / n if n else 0.0
    decline_rate = sum(r["declined"] for r in rows) / n if n else 0.0
    summary = {
        "study": manifest["study"], "mode": manifest["mode"], "git_at_run": manifest["git"],
        "episodes": {"scheduled_main": manifest["scheduled_episodes"]["main"], "scored": n,
                     "per_stratum": {s: len(rs) for s, rs in strata.items()}},
        "smoke_passed": manifest["smoke_passed"], "stopped": manifest["stopped"],
        "stop_reason": manifest["stop_reason"], "harness_error": manifest["harness_error"],
        "estimated_spend_usd": manifest["estimated_spend_usd"], "s1_check_sha256": reused,
        "bootstrap_seed": BOOTSTRAP_SEED, "margin_c2": MARGIN,
        "means": {"all": {c: _mean(r[c] for r in rows) for c in ("s0", "s1", "s3fr", "s3ao")},
                  **{s: {c: _mean(r[c] for r in rs) for c in ("s0", "s1", "s3fr", "s3ao")} for s, rs in strata.items()}},
        "C1_s3fr_minus_s1": c1, "C2_within_s3fr_minus_s1": c2, "C3_beyond_s3fr_minus_s1": c3,
        "C4_s3ao_minus_s3fr": c4, "C4_reading": {k: read_h(v) for k, v in c4.items()},
        "s3fr_minus_s0": paired(rows, "s3fr", "s0"), "gain_share_by_stratum": gain_share,
        "mechanism": {"rescue_rate_beyond": _mean(r["rescued"] for r in strata["beyond"]), "rescued": len(rescued),
                      "landing_rate_forced_rerank": _mean(r["landed_fr"] for r in rescued),
                      "landing_rate_agent": _mean(r["landed_ao"] for r in rescued),
                      "harm_rate_within": _mean(r["lost_fr"] for r in strata["within"])},
        "behaviour": behaviour, "cost_latency": cost, "tune_to_check": tune_to_check,
        "sensitivity": {"c1_excluding_declined": paired([r for r in rows if not r["declined"]], "s3fr", "s1")},
        "errors": {"declined_episodes": sum(r["declined"] for r in rows), "decline_rate": decline_rate,
                   "other_error_episodes": sum(r["other_error"] for r in rows), "other_error_rate": other_rate},
    }
    summary["outcome_rules"] = classify(paid=paid, manifest=manifest, other_error_rate=other_rate,
                                        decline_rate=decline_rate, tool_bugs=rejected.get("tool_bug", 0),
                                        c1=c1, c2=c2)
    return summary, rows


def _f(value, spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value, spec)


def report(s: dict) -> str:
    def ci(c: dict | None) -> str:
        return "n/a" if c is None else f"{c['mean']:+.3f} [{c['low']:+.3f}, {c['high']:+.3f}] (n = {c['n']})"

    m, b, c, e, t = s["mechanism"], s["behaviour"], s["cost_latency"], s["episodes"], s["tune_to_check"]
    lines = [
        f"# S3 confirmation on the check half ({s['mode']})", "",
        f"Episodes scored: {e['scored']}/{e['scheduled_main']} ({e['per_stratum']}); smoke passed: "
        f"{s['smoke_passed']}; stopped: {s['stopped']} ({s['stop_reason']}); estimated spend "
        f"${s['estimated_spend_usd']:.4f}. Outcome rules: {s['outcome_rules']}.", "",
        "Blind for S3; settings frozen at the pilot's; S1 reused from the S1 check study; one episode per query.", "",
        "## nDCG@10 means", "", "| queries | S0 | S1 | S3 + forced rerank | S3 + agent ordering |", "|---|---|---|---|---|",
        *[f"| {g} | {_f(v['s0'])} | {_f(v['s1'])} | {_f(v['s3fr'])} | {_f(v['s3ao'])} |" for g, v in s["means"].items()],
        "", "## Hypotheses (95% CI)", "",
        f"- C1 all check queries, S3+FR − S1: {ci(s['C1_s3fr_minus_s1'])}",
        f"- C2 within 20, S3+FR − S1 (margin {s['margin_c2']}): {ci(s['C2_within_s3fr_minus_s1'])}",
        f"- C3 beyond 20, S3+FR − S1: {ci(s['C3_beyond_s3fr_minus_s1'])}",
        f"- C4 S3+AO − S3+FR: overall {ci(s['C4_s3ao_minus_s3fr']['overall'])}, beyond "
        f"{ci(s['C4_s3ao_minus_s3fr']['beyond'])}, within {ci(s['C4_s3ao_minus_s3fr']['within'])} → {s['C4_reading']}",
        f"- S3+FR − S0: {ci(s['s3fr_minus_s0'])}; share of the gain over S1 by stratum: {s['gain_share_by_stratum']}",
        f"- tune → check: pilot reweighted to check proportions {ci(t['pilot_tune_reweighted_to_check'])}, check "
        f"{ci(t['check'])} ({t['note']})", "",
        "## Mechanism (labels, analysis only)", "",
        f"- rescue rate (beyond: a model search added a relevant document): {_f(m['rescue_rate_beyond'])} "
        f"({m['rescued']} queries); landing in the top 10: forced rerank {_f(m['landing_rate_forced_rerank'])}, "
        f"agent {_f(m['landing_rate_agent'])}",
        f"- harm (within: a relevant document in S1's top 10 lost): {_f(m['harm_rate_within'])}", "",
        "## Model behaviour (reported separately from harness settings)", "",
        *[f"- {st}: {b[st]}" for st in ("beyond", "within")],
        f"- stop reasons {b['stop_reasons']}; agent ranking {b['agent_ranking_status']}; rerank "
        f"{b['rerank_status']}; rejected tool calls {b['rejected_tool_calls']}", "",
        "## Cost and latency (estimates)", "",
        f"- $ per query: S3 {_f(c['s3_per_query_usd'], '.5f')} vs S1 {_f(c['s1_per_query_usd'], '.5f')}; per stratum "
        f"{c['per_stratum_usd']}; extra $ per rescued query {_f(c['extra_usd_per_rescued_query'], '.4f')}; rerank "
        f"share {_f(c['rerank_share_of_cost'])}; unpriced calls {c['unpriced_calls']}",
        f"- s per query: S3 {_f(c['s3_wall_s_per_query'], '.1f')} vs S1 {_f(c['s1_latency_s_per_query'], '.1f')}", "",
        "## Sensitivity", "",
        f"- C1 excluding declined episodes: {ci(s['sensitivity']['c1_excluding_declined'])}",
        f"- errors: {s['errors']}",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--s1-check", type=Path, default=S1_CHECK)
    parser.add_argument("--pilot-summary", type=Path, default=PILOT_SUMMARY)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)

    summary, rows = analyze(args.results, args.s1_check, args.data_dir, args.pilot_summary)
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
