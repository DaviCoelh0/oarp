# R2a: reranking headroom and gate ceilings on SciFact (development queries)

A free analysis on 2026-10-23, done before any model reranks anything. **No model
calls, $0.** It asks how much a reranking step *could* gain, how much of that a
gate could keep while reranking only some queries, and roughly what the calls
would cost. Every quality number here uses a **perfect** reranker or gate built
from the dev relevance labels, so these are upper bounds, not results of a
model.

## Setup

- Data, split and BM25: as in the [R1 baseline](../retrieval_baseline/README.md)
  (BEIR SciFact, 809 train queries used for development, test split not loaded,
  `bm25s` 0.3.13 with library defaults).
- Dev partition: the 809 queries are split into **tune** (405) and **check**
  (404) by SHA-256 of a fixed seed plus the query id
  (`oarp.envs.retrieval.data.dev_partition`). No labels are used to split.
  Settings are chosen on tune and confirmed on check. The test queries stay
  held out.
- Strategies (scored offline, `oarp.envs.retrieval.strategies`):
  - **S0**: BM25 order.
  - **S1**: rerank BM25's top k = 20 for every query.
  - **S2**: rule gate, which reranks when the relative gap (s1 − s2) / s1
    between BM25's top two scores is below τ (s1 = 0 counts as "rerank").
  - **S2b**: model gate, which reranks when a model does not judge rank 1
    relevant. It is not run here, because it needs model calls.
- Oracle reranker: relevant documents within the top k are moved to the
  front, in BM25 order within a grade. Oracle gate: rerank exactly when rank 1
  is not relevant.

Reproduce: `pip install -e ".[retrieval]"`, download SciFact
(`python -m oarp.envs.retrieval.data`), then
`venv/bin/python experiments/retrieval_rerank_headroom/run_analysis.py`.

## Results (all 809 dev queries unless noted)

**1. Ceiling of a perfect reranker of BM25's top k** (BM25 alone: nDCG@10 0.667)

| k | first relevant in top k | nDCG@10 | Recall@10 |
|---|---|---|---|
| 10 | 80.2% | 0.791 | 0.787 |
| 20 | 85.4% | 0.845 | 0.842 |
| 30 | 87.3% | 0.863 | 0.861 |
| 50 | 88.3% | 0.876 | 0.874 |
| 100 | 91.2% | 0.908 | 0.906 |

**2. Oracle gate with a perfect top-20 reranker.** It reranks 44.1% of queries
and reaches nDCG@10 0.841, against 0.845 for reranking every query. A perfect
gate would keep almost all of the gain at less than half the reranking calls.
It loses a little where rank 1 is relevant but other relevant documents sit
lower.

**3. Rule gate S2 with a perfect reranker** (τ matched on tune to a target
rerank rate, reported on check; check S0 0.666, S1 0.851, oracle gate 0.847 at
rate 0.448):

| target rate | τ | check rerank rate | check nDCG@10 |
|---|---|---|---|
| 0.20 | 0.051 | 0.193 | 0.729 |
| 0.40 | 0.111 | 0.342 | 0.767 |
| 0.435 (oracle gate's rate on tune) | 0.134 | 0.408 | 0.784 |
| 0.50 | 0.177 | 0.495 | 0.804 |
| 0.60 | 0.257 | 0.629 | 0.829 |
| 0.80 | 0.457 | 0.849 | 0.849 |

At about the oracle gate's rerank rate, the rule gate keeps about 65% of the
possible gain on check (0.784 vs 0.847, with 0.666 meaning no gain). It needs
a rate of about 0.8 to match always-rerank.

**4. How well BM25 signals predict "rank 1 not relevant"** (AUC; lower signal
= predicted miss; 0.5 = chance):

| signal | all | tune | check |
|---|---|---|---|
| relative gap (s1 − s2) / s1 | 0.833 | 0.846 | 0.819 |
| raw gap s1 − s2 | 0.847 | 0.852 | 0.840 |
| top score s1 | 0.755 | 0.751 | 0.761 |

**5. Estimated cost per query** (Claude Haiku 5.5 at $0.10 / $0.50 per million
input / output tokens, prices re-checked against the official pricing page on
2026-10-23). Input tokens come from the real prompts at about 4 characters per
token, a heuristic. Output tokens are **assumptions**: the visible answer plus
adaptive thinking, rerank 60–560 tokens at low effort and 260–2,060 at medium,
gate 5–205 and 105–1,005. Neither is measured yet.

| call | est. input tokens | low effort | medium effort |
|---|---|---|---|
| rerank top 20 | ~7,600 | $0.0008–0.0010 | $0.0009–0.0018 |
| gate (top 3) | ~1,200 | $0.0001–0.0002 | $0.0002–0.0006 |

A gate saves money only if gate cost / rerank cost < 1 − its rerank rate. At
the oracle rate (0.44) that bound is 0.56. The estimated ratio is 0.12–0.29 at
low effort and 0.10–0.70 at medium. At medium effort, the gate's own thinking
could cancel the saving.

`results/summary.json` and `results/analysis.md` hold the aggregates.
Per-query signals and ranks are derived from the labels and stay in a
gitignored `results/private/` folder.

## Interpretation and limitations

- **Upper bounds only.** No model has reranked or gated anything yet. A real
  reranker will land below the perfect one, and its cost is an estimate until
  measured.
- **Gating has room.** A perfect gate needs fewer than half the rerank calls
  for almost the full gain. A free BM25 signal predicts the rank-1 misses well
  (AUC ≈ 0.83–0.85) but recovers only part of that at the same rate. A model
  gate (S2b) has to beat this rule at an equal rerank rate, and cost less than
  the reranks it saves.
- **The raw gap predicts slightly better than the relative gap** (0.847 vs
  0.833 on all dev queries). S2 is defined with the relative gap, as decided
  before this analysis. The difference is small and comes from one dataset.
- **Sparse labels** (1.14 relevant documents per query): per-query values are
  coarse. Tune and check differ by up to 0.03 in AUC, which shows the noise
  level at about 400 queries.
- **Contamination:** SciFact is public, and a hosted model may have seen it.
  That cannot be ruled out for later model-based stages.
