"""R2a: free reranking-headroom analysis on SciFact's development (train) queries.

    venv/bin/python experiments/retrieval_rerank_headroom/run_analysis.py   # free: no model calls

Model-free and deterministic; the test split is never loaded. Uses dev labels,
which may guide selection. Reports, over all dev queries and over the fixed
tune / check halves (`dev_partition`):

1. the perfect-reranker ceiling for reranking BM25's top k (oracle by k);
2. the oracle gate (rerank only when rank 1 is not relevant) with the perfect
   reranker: an upper bound for any gate, next to S0 and S1-with-oracle;
3. the S2 rule gate (relative gap below tau) with the perfect reranker, tau
   matched on tune to a target rerank rate and reported on check;
4. how well each BM25 signal (relative gap, raw gap, top score) predicts
   "rank 1 not relevant" (AUC; low signal = predicted miss);
5. ESTIMATED cost per query of a rerank call and a gate call: input tokens
   from the real prompts at ~4 characters per token (a heuristic), output
   tokens from stated assumptions (unmeasured), priced with pricing.py.

Writes results/summary.json and results/analysis.md (aggregates only) and
results/private/per_query.jsonl (label-derived, gitignored).
"""

import argparse
import json
import math
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from oarp.core.pricing import PRICES_AS_OF, estimate_cost  # noqa: E402
from oarp.core.trace import git_info  # noqa: E402
from oarp.core.types import Usage  # noqa: E402
from oarp.envs.retrieval import gate as gate_mod  # noqa: E402
from oarp.envs.retrieval import rerank as rerank_mod  # noqa: E402
from oarp.envs.retrieval.analysis import mixed_mean, oracle_gate, oracle_rerank, rank1_relevant  # noqa: E402
from oarp.envs.retrieval.bm25 import BM25Config, BM25Index, provenance  # noqa: E402
from oarp.envs.retrieval.data import DEFAULT_DATA_DIR, DEV_PARTITION_SEED, dev_partition, load_scifact  # noqa: E402
from oarp.envs.retrieval.metrics import auc, first_relevant_rank, mrr_at_k, ndcg_at_k, recall_at_k  # noqa: E402
from oarp.envs.retrieval.strategies import bm25_signals, rule_gate, tau_for_rate  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
DEPTHS = (10, 20, 30, 50, 100)
TARGET_RATES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.8)
SIGNALS = ("relative_gap", "raw_gap", "top_score")
MODEL = "claude-haiku-5-5"
CHARS_PER_TOKEN = 4  # heuristic; the S1 pilot measures real token counts
# Output-token ASSUMPTIONS (visible answer + adaptive thinking), not measurements.
OUTPUT_TOKENS = {
    "rerank": {"low": (60, 560), "medium": (260, 2060)},  # ~3 tokens per number for k = 20, plus thinking
    "gate": {"low": (5, 205), "medium": (105, 1005)},
}


def metric_row(ranking: list[str], judged: dict[str, int]) -> dict[str, float]:
    return {"ndcg@10": ndcg_at_k(ranking, judged, 10), "recall@10": recall_at_k(ranking, judged, 10),
            "mrr@10": mrr_at_k(ranking, judged, 10)}


def means(rows: dict[str, dict[str, float]], qids: list[str]) -> dict[str, float]:
    return {m: statistics.fmean(rows[q][m] for q in qids) for m in ("ndcg@10", "recall@10", "mrr@10")}


def evaluate(data_dir: Path, config: BM25Config = BM25Config(),
             rerank_config: rerank_mod.RerankConfig = rerank_mod.RerankConfig()) -> tuple[dict, list[dict]]:
    split = load_scifact(data_dir, "train")
    index = BM25Index(split.corpus, config)
    qids = sorted(split.queries)
    halves = dev_partition(qids)
    groups = {"all": qids, **halves}
    scored = {q: index.rank(split.queries[q]) for q in qids}
    bm25 = {q: [d for d, _ in scored[q]] for q in qids}
    qrels = split.qrels
    signals = {q: bm25_signals(scored[q]) for q in qids}
    miss = {q: not rank1_relevant(bm25[q], qrels[q]) for q in qids}
    k = rerank_config.k

    # 1. Oracle by k
    s0 = {q: metric_row(bm25[q], qrels[q]) for q in qids}
    oracle_by_k = {}
    oracle_rows = {}
    for depth in DEPTHS:
        rows = {q: metric_row(oracle_rerank(bm25[q], qrels[q], depth), qrels[q]) for q in qids}
        oracle_rows[depth] = rows
        oracle_by_k[str(depth)] = {
            g: {**means(rows, ids), "first_relevant_in_top_k": statistics.fmean(
                (first_relevant_rank(bm25[q], qrels[q]) or math.inf) <= depth for q in ids)}
            for g, ids in groups.items()}

    # 2. Oracle gate with the perfect reranker at depth k
    s1_rows = oracle_rows[k]
    decisions = oracle_gate(bm25, qrels)
    gated_rows = {q: s1_rows[q] if decisions[q] else s0[q] for q in qids}
    oracle_gate_result = {g: {"rerank_rate": statistics.fmean(decisions[q] for q in ids),
                              "S0": means(s0, ids), "S1_oracle_reranker": means(s1_rows, ids),
                              "oracle_gate_oracle_reranker": means(gated_rows, ids)}
                          for g, ids in groups.items()}

    # 3. S2 rule gate (relative gap) with the perfect reranker: tau matched on tune, reported on check
    s0_ndcg = {q: s0[q]["ndcg@10"] for q in qids}
    s1_ndcg = {q: s1_rows[q]["ndcg@10"] for q in qids}
    gaps = {q: signals[q]["relative_gap"] for q in qids}
    oracle_rate_tune = oracle_gate_result["tune"]["rerank_rate"]
    rule_curve = []
    for target in (*TARGET_RATES, oracle_rate_tune):
        tau = tau_for_rate([gaps[q] for q in halves["tune"]], target)
        flags = rule_gate({q: gaps[q] for q in halves["check"]}, tau)
        rule_curve.append({"target_rate": round(target, 4), "tau": tau,
                           "check_rerank_rate": statistics.fmean(flags.values()),
                           "check_ndcg@10": mixed_mean(flags, s0_ndcg, s1_ndcg),
                           "matches_oracle_rate": target == oracle_rate_tune})

    # 4. Signal AUCs for "rank 1 not relevant" (low signal = predicted miss)
    def auc_or_none(scores: list[float], labels: list[bool]) -> float | None:
        return auc(scores, labels) if 0 < sum(labels) < len(labels) else None  # needs both classes

    signal_auc = {s: {g: auc_or_none([-signals[q][s] for q in ids], [miss[q] for q in ids])
                      for g, ids in groups.items()} for s in SIGNALS}

    # 5. Estimated cost per query
    def prompt_chars(messages) -> int:
        return sum(len(m.content) for m in messages)

    rerank_chars = [prompt_chars(rerank_mod.rerank_messages(
        split.queries[q], [split.corpus[d] for d in bm25[q][:k]], rerank_config)) for q in qids]
    gate_chars = [prompt_chars(gate_mod.gate_messages(
        split.queries[q], [split.corpus[d] for d in bm25[q][:gate_mod.GATE_TOP]])) for q in qids]
    cost = {}
    for call, chars in (("rerank", rerank_chars), ("gate", gate_chars)):
        input_tokens = round(statistics.fmean(chars) / CHARS_PER_TOKEN)
        cost[call] = {"mean_prompt_chars": round(statistics.fmean(chars)), "est_input_tokens": input_tokens,
                      "per_query_usd": {effort: [estimate_cost(MODEL, Usage(input_tokens, out)) for out in outs]
                                        for effort, outs in OUTPUT_TOKENS[call].items()}}
    for effort in ("low", "medium"):
        lo_r, hi_r = cost["rerank"]["per_query_usd"][effort]
        lo_g, hi_g = cost["gate"]["per_query_usd"][effort]
        cost.setdefault("gate_over_rerank", {})[effort] = [lo_g / hi_r, hi_g / lo_r]
    # Gate pays off iff gate_cost + rate * rerank_cost < rerank_cost, i.e. gate/rerank < 1 - rate.
    cost["break_even_note"] = ("a gate saves money iff gate cost / rerank cost < 1 - its rerank rate; "
                               f"the oracle gate's rate on all dev queries is "
                               f"{oracle_gate_result['all']['rerank_rate']:.3f}")

    per_query = [{"query_id": q, "half": "tune" if q in set(halves["tune"]) else "check", **signals[q],
                  "rank1_relevant": not miss[q], "first_relevant_rank": first_relevant_rank(bm25[q], qrels[q])}
                 for q in qids]
    summary = {
        "study": "R2a: free reranking-headroom analysis on SciFact train queries (no model calls)",
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_info(REPO), "python": platform.python_version(), **provenance(config),
        **rerank_mod.provenance(rerank_config), **gate_mod.provenance(),
        "split": "train", "queries": {g: len(ids) for g, ids in groups.items()},
        "dev_partition_seed": DEV_PARTITION_SEED,
        "rerank_k": k, "oracle_by_k": oracle_by_k, "oracle_gate": oracle_gate_result,
        "rule_gate_relative_gap_oracle_reranker": rule_curve, "signal_auc_rank1_not_relevant": signal_auc,
        "estimated_cost": {"model": MODEL, "prices_as_of": PRICES_AS_OF, "chars_per_token": CHARS_PER_TOKEN,
                           "output_token_assumptions": OUTPUT_TOKENS, **cost},
    }
    return summary, per_query


def report(s: dict) -> str:
    n = s["queries"]
    lines = [
        "# R2a: reranking headroom on SciFact (train queries)", "",
        f"Model-free, deterministic, $0. Queries: {n['all']} (tune {n['tune']}, check {n['check']}). "
        "Test split not loaded. All numbers below use a PERFECT reranker or gate (they read labels): "
        "upper bounds, not results of any model.", "",
        "## 1. Perfect reranker of BM25's top k", "",
        "| k | first relevant in top k | nDCG@10 | Recall@10 | MRR@10 |", "|---|---|---|---|---|",
        *[f"| {k} | {v['all']['first_relevant_in_top_k']:.3f} | {v['all']['ndcg@10']:.3f} | "
          f"{v['all']['recall@10']:.3f} | {v['all']['mrr@10']:.3f} |" for k, v in s["oracle_by_k"].items()], "",
        f"## 2. Oracle gate (rerank only when rank 1 is not relevant), perfect reranker, k = {s['rerank_k']}", "",
        "| queries | rerank rate | S0 nDCG@10 | S1 (perfect) nDCG@10 | oracle gate nDCG@10 |", "|---|---|---|---|---|",
        *[f"| {g} | {v['rerank_rate']:.3f} | {v['S0']['ndcg@10']:.3f} | {v['S1_oracle_reranker']['ndcg@10']:.3f} | "
          f"{v['oracle_gate_oracle_reranker']['ndcg@10']:.3f} |" for g, v in s["oracle_gate"].items()], "",
        "## 3. Rule gate (relative gap < tau), perfect reranker; tau matched on tune, reported on check", "",
        "| target rate | tau | check rerank rate | check nDCG@10 |", "|---|---|---|---|",
        *[f"| {r['target_rate']:.3f}{' (oracle rate)' if r['matches_oracle_rate'] else ''} | {r['tau']:.4f} | "
          f"{r['check_rerank_rate']:.3f} | {r['check_ndcg@10']:.3f} |"
          for r in s["rule_gate_relative_gap_oracle_reranker"]], "",
        "## 4. Does a BM25 signal predict \"rank 1 not relevant\"? (AUC, 0.5 = chance)", "",
        "| signal | all | tune | check |", "|---|---|---|---|",
        *[f"| {sig} | " + " | ".join("n/a" if v[g] is None else f"{v[g]:.3f}" for g in ("all", "tune", "check")) + " |"
          for sig, v in s["signal_auc_rank1_not_relevant"].items()], "",
        "## 5. Estimated cost per query (Haiku 5.5; estimates, not measurements)", "",
    ]
    c = s["estimated_cost"]
    lines += ["| call | mean prompt chars | est. input tokens | low effort $ | medium effort $ |", "|---|---|---|---|---|"]
    for call in ("rerank", "gate"):
        v = c[call]
        lo, med = v["per_query_usd"]["low"], v["per_query_usd"]["medium"]
        lines.append(f"| {call} | {v['mean_prompt_chars']} | {v['est_input_tokens']} | "
                     f"{lo[0]:.5f}-{lo[1]:.5f} | {med[0]:.5f}-{med[1]:.5f} |")
    lines += ["", f"Gate cost / rerank cost: low {c['gate_over_rerank']['low'][0]:.2f}-"
              f"{c['gate_over_rerank']['low'][1]:.2f}, medium {c['gate_over_rerank']['medium'][0]:.2f}-"
              f"{c['gate_over_rerank']['medium'][1]:.2f}. {c['break_even_note']}.",
              f"Input tokens at ~{c['chars_per_token']} characters per token (heuristic); output tokens are "
              f"assumptions: {c['output_token_assumptions']}. Prices as of {c['prices_as_of']}."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args(argv)

    summary, rows = evaluate(args.data_dir)
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
