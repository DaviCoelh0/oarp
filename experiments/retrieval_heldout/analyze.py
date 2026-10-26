"""Final held-out evaluation, free part: S0, S1, S3 and the frozen policy P* on
every test query, as the approved contract pre-registers.

    venv/bin/python experiments/retrieval_heldout/analyze.py

Reads this study's records (main phase only; smoke queries are excluded) and
the configuration search's committed summary. P* is frozen here, before the
test split is loaded: search (S3) if top_score < P_STAR["search_tau"], else
rerank (S1) if raw_gap < P_STAR["rerank_tau"], else BM25 (S0), routed with the
search's own `route` (imported, pinned) and checked against its recorded
selection. The test split is loaded only when the records came from a paid
run on it (`include_heldout=True`); a dry run is analysed on its dev stand-in.

- H1 S1 - S0 > 0; H2 S3 - S1 > 0; H3 P* - S1 > 0 (primary); H4 P* - S3 >
  -0.01 at a lower cost than S3; the optimisation claim needs H3 and H4;
- paired percentile bootstrap over queries, 10,000 resamples, seed 20261030;
- descriptive: strata from test labels after the run (first relevant within
  BM25's top 20 or not), cost, latency, P*'s routing, and the dev Pareto
  frontier's policies evaluated on test (no re-selection on test).

Writes results/summary.json and results/analysis.md (aggregates only) and
results/private/per_query.jsonl (label-derived, gitignored).
"""

import argparse
import importlib.util
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from oarp.envs.retrieval import study  # noqa: E402
from oarp.envs.retrieval.bm25 import BM25Config, BM25Index  # noqa: E402
from oarp.envs.retrieval.data import DEFAULT_DATA_DIR, dev_partition, load_scifact  # noqa: E402
from oarp.envs.retrieval.metrics import first_relevant_rank, ndcg_at_k  # noqa: E402
from oarp.envs.retrieval.stats import paired_bootstrap_ci  # noqa: E402
from oarp.envs.retrieval.strategies import bm25_signals  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SEARCH_DIR = REPO / "experiments" / "retrieval_config_search"
SEARCH_SHA256 = "e8867a5d20d225328ab0b2cc76c16b64b86a97f6da6e4e1b4bae3229dbf7d578"
SEARCH_SUMMARY_SHA256 = "7d174888a16f4ff12c038ad7c6b99566a91b47a3dbaa253f0f95a32df374cc27"
P_STAR = {"search_signal": "top_score", "search_tau": 9.850902557373047, "search_rate": 0.3,
          "rerank_signal": "raw_gap", "rerank_tau": 7.0107831954956055, "rerank_rate": 0.8}
BOOTSTRAP_SEED = 20261030
H4_MARGIN = -0.01
DECLINE = "ModelRefused"
MAX_OTHER_ERROR_RATE = 0.05
DECLINE_WARNING_RATE = 0.10
DEPTH = 20


def load_search():
    spec = importlib.util.spec_from_file_location("config_search_for_heldout", SEARCH_DIR / "search.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cs = load_search()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _mean(values) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def _tau(value) -> float:
    return {"+inf": math.inf, "-inf": -math.inf}.get(value, value) if isinstance(value, str) else value


def check_frozen_policy(summary_path: Path, *, pinned: bool) -> dict:
    """The configuration search's committed selection must equal P_STAR (and, pinned, its files their hashes)."""
    hashes = {"search.py": study.sha256_file(SEARCH_DIR / "search.py"), "summary.json": study.sha256_file(summary_path)}
    if pinned and (hashes["search.py"] != SEARCH_SHA256 or hashes["summary.json"] != SEARCH_SUMMARY_SHA256):
        raise SystemExit("refusing: the configuration search files differ from their pinned SHA-256")
    selected = json.loads(summary_path.read_text())["selected"]
    if {k: _tau(selected[k]) for k in P_STAR} != P_STAR:
        raise SystemExit("refusing: P* differs from the configuration search's recorded selection")
    return hashes


def read(ci: dict | None, bound: float = 0.0) -> bool | None:
    return None if ci is None else ci["low"] > bound


def classify(*, paid: bool, manifest: dict, other_error_rate: float, decline_rate: float,
             h: dict[str, dict | None], cost_lower: bool) -> dict[str, object]:
    if not paid:
        run = "dry run: no outcome"
    elif manifest["harness_error"]:
        run = "invalid (harness error)"
    elif not manifest["smoke_passed"]:
        run = "stopped after the smoke phase (no result claimed)"
    elif manifest["stopped"] or manifest["completed"]["main"] < manifest["scheduled"]["main"]:
        run = "incomplete (no hypothesis claimed)"
    elif other_error_rate > MAX_OTHER_ERROR_RATE:
        run = "invalid (non-decline model errors above 5%)"
    else:
        run = "valid"
    if run != "valid":
        return {"run": run, "decline_warning": decline_rate > DECLINE_WARNING_RATE, "H1": None, "H2": None,
                "H3": None, "H4": None, "optimisation_claim": None}
    h4 = bool(read(h["H4"], H4_MARGIN)) and cost_lower
    out = {"run": run, "decline_warning": decline_rate > DECLINE_WARNING_RATE,
           "H1": read(h["H1"]), "H2": read(h["H2"]), "H3": read(h["H3"]), "H4": h4}
    out["optimisation_claim"] = bool(out["H3"]) and h4
    return out


def analyze(results: Path, data_dir: Path, summary_path: Path = SEARCH_DIR / "results" / "summary.json",
            *, pinned: bool | None = None) -> tuple[dict, list[dict]]:
    manifest = json.loads((results / "manifest.json").read_text())
    paid = manifest["mode"] == "paid"
    search_hashes = check_frozen_policy(summary_path, pinned=paid if pinned is None else pinned)
    dev = load_scifact(data_dir, "train")
    if manifest["query_split"] == "test":
        if not paid and pinned is not False:
            raise SystemExit("refusing: only a paid run's records may open the real test split")
        split = load_scifact(data_dir, "test", include_heldout=True)
    else:
        check = set(dev_partition(sorted(dev.queries))["check"])
        split = dev
        split_queries = {q: dev.queries[q] for q in check}
    queries = split.queries if manifest["query_split"] == "test" else split_queries
    qrels = split.qrels
    records = [r for r in _rows(results / "private" / "records.jsonl") if r["phase"] == "main"]
    if any(r["query_id"] not in queries for r in records):
        raise SystemExit("refusing: a record's query is not in the evaluated split")
    index = BM25Index(dev.corpus, BM25Config())
    priced_s1 = [r["s1"]["estimated_usd"] for r in records if r["s1"]["estimated_usd"] is not None]
    s1_fill = _mean(priced_s1) or 0.0

    rows = []
    for r in records:
        q, judged = r["query_id"], qrels[r["query_id"]]
        scored = index.rank(queries[q])
        bm25 = [d for d, _ in scored]
        s1, s3 = r["s1"], r["s3"]
        if s1["bm25_top"] != bm25[:DEPTH] or s3["candidates"][:DEPTH] != bm25[:DEPTH]:
            raise SystemExit("refusing: a recorded BM25 top 20 differs from a fresh BM25 ranking")
        s3_calls = s3["call_usd"]["agent"] + s3["call_usd"]["rerank"]
        rr = s3["rerank"] or {}
        rank = first_relevant_rank(bm25, judged)
        row = {"query_id": q, "stratum": "within" if rank is not None and rank <= DEPTH else "beyond",
               **bm25_signals(scored),
               "s0": ndcg_at_k(bm25, judged, 10), "s0_usd": 0.0, "s0_s": 0.0,
               "s1": ndcg_at_k(s1["ranking_top"] + bm25[DEPTH:], judged, 10),
               "s1_usd": s1["estimated_usd"] if s1["estimated_usd"] is not None else s1_fill,
               "s1_s": s1["latency_s"],
               "s3": ndcg_at_k(s3["ranking"], judged, 10), "s3_usd": sum(c for c in s3_calls if c is not None),
               "s3_s": s3["wall_s"],
               "declined": s1["error_kind"] == DECLINE or (s3["error"] or "").startswith(DECLINE)
               or rr.get("error_kind") == DECLINE,
               "other_error": (s1["status"] == "model_error" and s1["error_kind"] != DECLINE)
               or (s3["stop_reason"] == "model_error" and not (s3["error"] or "").startswith(DECLINE))
               or (rr.get("status") == "model_error" and rr.get("error_kind") != DECLINE),
               "s3_searches": s3["tool_calls"]["executed_searches"],
               "unpriced": (s1["estimated_usd"] is None) + sum(c is None for c in s3_calls)}
        row["route"] = cs.route(row, P_STAR)
        row["pstar"], row["pstar_usd"], row["pstar_s"] = (row[row["route"]], row[f"{row['route']}_usd"],
                                                          row[f"{row['route']}_s"])
        rows.append(row)

    def ci(rs: list[dict], a: str, b: str) -> dict | None:
        return paired_bootstrap_ci([r[a] - r[b] for r in rs], seed=BOOTSTRAP_SEED) if rs else None

    h = {"H1": ci(rows, "s1", "s0"), "H2": ci(rows, "s3", "s1"), "H3": ci(rows, "pstar", "s1"),
         "H4": ci(rows, "pstar", "s3")}
    cost = {c: _mean(r[f"{c}_usd"] for r in rows) for c in ("s0", "s1", "s3", "pstar")}
    latency = {c: _mean(r[f"{c}_s"] for r in rows) for c in ("s0", "s1", "s3", "pstar")}
    strata = {s: [r for r in rows if r["stratum"] == s] for s in ("beyond", "within")}
    frontier = []
    for p in json.loads(summary_path.read_text())["pareto_frontier_check"]:
        policy = {k: _tau(p[k]) for k in P_STAR}
        frontier.append({**policy, "check_ndcg": p["ndcg"], "check_usd": p["usd"],
                         **{f"test_{k}": v for k, v in cs.score(rows, policy).items()}} if rows else policy)
    n = len(rows)
    other_rate = sum(r["other_error"] for r in rows) / n if n else 0.0
    decline_rate = sum(r["declined"] for r in rows) / n if n else 0.0
    summary = {
        "study": manifest["study"], "mode": manifest["mode"], "query_split": manifest["query_split"],
        "git_at_run": manifest["git"], "search_sha256": search_hashes, "p_star": P_STAR,
        "bootstrap_seed": BOOTSTRAP_SEED, "h4_margin": H4_MARGIN,
        "queries": {"scheduled": manifest["scheduled"]["main"], "scored": n,
                    "per_stratum": {s: len(rs) for s, rs in strata.items()}},
        "smoke_passed": manifest["smoke_passed"], "stopped": manifest["stopped"], "stop_reason": manifest["stop_reason"],
        "harness_error": manifest["harness_error"], "estimated_spend_usd": manifest["estimated_spend_usd"],
        "means": {"all": {c: _mean(r[c] for r in rows) for c in ("s0", "s1", "s3", "pstar")},
                  **{s: {c: _mean(r[c] for r in rs) for c in ("s0", "s1", "s3", "pstar")} for s, rs in strata.items()}},
        **h,
        "by_stratum": {s: {"s3_minus_s1": ci(rs, "s3", "s1"), "pstar_minus_s1": ci(rs, "pstar", "s1"),
                           "pstar_minus_s3": ci(rs, "pstar", "s3")} for s, rs in strata.items()},
        "cost_usd_per_query": cost, "latency_s_per_query": latency,
        "pstar_routing": dict(Counter(r["route"] for r in rows)),
        "s3_searches": dict(sorted(Counter(r["s3_searches"] for r in rows).items())),
        "frontier_on_test": frontier,
        "errors": {"declined_queries": sum(r["declined"] for r in rows), "decline_rate": decline_rate,
                   "other_error_queries": sum(r["other_error"] for r in rows), "other_error_rate": other_rate,
                   "unpriced_calls": sum(r["unpriced"] for r in rows), "s1_unpriced_filled_usd": s1_fill},
        "sensitivity": {"excluding_declined": {k: ci([r for r in rows if not r["declined"]], a, b)
                                               for k, (a, b) in {"H1": ("s1", "s0"), "H2": ("s3", "s1"),
                                                                 "H3": ("pstar", "s1"), "H4": ("pstar", "s3")}.items()}},
    }
    summary["outcome_rules"] = classify(paid=paid, manifest=manifest, other_error_rate=other_rate,
                                        decline_rate=decline_rate, h=h,
                                        cost_lower=cost["pstar"] is not None and cost["pstar"] < cost["s3"])
    return summary, rows


def _f(value, spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value, spec)


def report(s: dict) -> str:
    def ci(c: dict | None) -> str:
        return "n/a" if c is None else f"{c['mean']:+.3f} [{c['low']:+.3f}, {c['high']:+.3f}] (n = {c['n']})"

    m, c, t = s["means"]["all"], s["cost_usd_per_query"], s["latency_s_per_query"]
    lines = [
        f"# Held-out evaluation ({s['mode']}; {s['query_split']})", "",
        f"Queries scored: {s['queries']['scored']}/{s['queries']['scheduled']} ({s['queries']['per_stratum']}); "
        f"smoke passed: {s['smoke_passed']}; stopped: {s['stopped']} ({s['stop_reason']}); estimated spend "
        f"${s['estimated_spend_usd']:.4f}. Outcome rules: {s['outcome_rules']}.", "",
        "| | S0 (BM25) | S1 (rerank) | S3 (search agent) | P* (searched policy) |", "|---|---|---|---|---|",
        f"| nDCG@10 | {_f(m['s0'])} | {_f(m['s1'])} | {_f(m['s3'])} | {_f(m['pstar'])} |",
        f"| $ per query | {_f(c['s0'], '.5f')} | {_f(c['s1'], '.5f')} | {_f(c['s3'], '.5f')} | {_f(c['pstar'], '.5f')} |",
        f"| s per query | {_f(t['s0'], '.1f')} | {_f(t['s1'], '.1f')} | {_f(t['s3'], '.1f')} | {_f(t['pstar'], '.1f')} |",
        "", "## Pre-registered comparisons (95% CI)", "",
        f"- H1 S1 − S0: {ci(s['H1'])}", f"- H2 S3 − S1: {ci(s['H2'])}",
        f"- H3 P* − S1 (primary): {ci(s['H3'])}", f"- H4 P* − S3 (margin {s['h4_margin']}): {ci(s['H4'])}", "",
        "## Descriptive", "",
        *[f"- {st}: means {s['means'][st]}; S3 − S1 {ci(v['s3_minus_s1'])}; P* − S1 {ci(v['pstar_minus_s1'])}; "
          f"P* − S3 {ci(v['pstar_minus_s3'])}" for st, v in s["by_stratum"].items()],
        f"- P* routing: {s['pstar_routing']}; S3 searches per episode: {s['s3_searches']}",
        f"- errors: {s['errors']}",
        f"- sensitivity, excluding declined queries: " + "; ".join(f"{k} {ci(v)}" for k, v in
                                                                  s["sensitivity"]["excluding_declined"].items()),
        "", "## The dev Pareto frontier's policies on test (chosen on check; no re-selection)", "",
        "| check $ | check nDCG@10 | test $ | test nDCG@10 |", "|---|---|---|---|",
        *[f"| {_f(p.get('check_usd'), '.5f')} | {_f(p.get('check_ndcg'))} | {_f(p.get('test_usd'), '.5f')} | "
          f"{_f(p.get('test_ndcg'))} |" for p in s["frontier_on_test"]],
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)
    summary, rows = analyze(args.results, args.data_dir)
    (args.results / "private").mkdir(parents=True, exist_ok=True)
    (args.results / "summary.json").write_text(json.dumps(cs._clean(summary), indent=2) + "\n")
    (args.results / "analysis.md").write_text(report(summary))
    with (args.results / "private" / "per_query.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(report(summary))
    return summary


if __name__ == "__main__":
    main()
