# S1 confirmation on the development check half

A paid study on 2026-10-24, run from commit `d6c4547` (clean tree, no untracked
files). **444 calls; $0.572 for the calls that returned usage, $0.608 counted
by the budget guard,** which counts each declined call at its $0.004
worst case. The design was fixed before the run (below). These are
development data, not a held-out result.

## Design (fixed before the run)

- **Question:** does the reranking gain measured in the
  [S1 pilot](../retrieval_s1_pilot/README.md) (on 100 tune queries) hold
  **with nothing changed** on the 404 queries of the check half of
  SciFact's train split? No setting had been chosen on these queries.
- **Condition:** S1-plain exactly as in the pilot: Claude Haiku 5.5 (effort
  `medium`, adaptive thinking, `max_tokens` 4096, no sampling parameters)
  reranks BM25's top 20 in one listwise call, with candidates shown in BM25
  order. The runner checked that the prompts and the reranker settings equal
  the ones recorded in the pilot's manifest.
- **Schedule:** all 404 queries once, then passes 2 and 3 on 20 of them
  (chosen by a seeded hash of the query id). 444 calls, sequential, standard
  API.
- **Primary analysis:** paired difference in nDCG@10 (S1-plain − BM25), with a
  paired bootstrap 95% CI over the 404 queries. A declined or failed query
  keeps the BM25 order. A sensitivity analysis excluding safety-declined
  queries was fixed in advance.
- **Rules fixed in advance:**
  - invalid if non-decline model errors exceed 5%;
  - a warning if declines exceed 10%;
  - confirmed / not confirmed / reversed by whether the CI is above, around
    or below 0;
  - a cost surprise above $0.0026 per call.

## Results

**Primary: S1-plain − BM25 = +0.120 nDCG@10, 95% CI [+0.096, +0.144]
(n = 404). Confirmed** by the pre-stated rule. The run is valid, with 0
non-decline model errors.

| | BM25 (S0) | S1-plain | perfect top-20 reranker |
|---|---|---|---|
| nDCG@10 | 0.666 | 0.786 | 0.851 |
| Recall@10 | 0.789 | 0.841 | |
| MRR@10 | 0.633 | 0.778 | |

- **Share of the possible gain:** 65% of what a perfect reranker of the same
  20 candidates would add.
- **Per query:** 113 improved, 14 got worse, 277 unchanged. Harm rate 3.5%,
  or 2.7% where BM25's rank 1 was already relevant.
- **By where BM25 put the first relevant document:**

  | first relevant at | queries | BM25 nDCG@10 | S1-plain nDCG@10 |
  |---|---|---|---|
  | rank 1 | 223 | 0.990 | 0.985 |
  | ranks 2–10 | 103 | 0.470 | 0.785 |
  | ranks 11–20 | 23 | 0.000 | 0.738 |
  | beyond 20 (unreachable) | 55 | 0.000 | 0.000 |

- **Declined requests:** 9 of 444 calls (2.0%) were declined by the
  provider's safety classifier, all category "bio": 8 in pass 1 and 1 in a
  stability pass. They were below the 10% warning level and were not
  retried. The pre-specified analysis excluding the 8 declined queries gives
  +0.122 [+0.098, +0.148] (n = 396).
- **Output format:** every answered call returned a valid full permutation
  (0 repairs, 0 truncations).
- **Stability** (20 queries × 3 passes): nDCG@10 varied across passes for 2
  of the 20 queries (mean per-query SD 0.004). Top-1 agreement was 75%, and
  the mean Kendall τ between passes on the top 20 was 0.76. In the pilot, 0
  of 20 varied, with 95% top-1 agreement. Outputs are stochastic, but the
  effect on quality was small here.
- **Cost and latency:** 10,845 input / 463 output tokens per call on average
  (p90 12,625 / 613), $0.00132 per call, median 4.3 s (p90 5.7 s). No cost
  surprise.

`results/summary.json` and `results/analysis.md` hold the aggregates,
including, for later gate studies, the free score-gap gate and the perfect
gate applied to these S1 rankings. Per-call records and per-query metrics
stay in a gitignored `results/private/` folder.

## Interpretation and limitations

- **The pilot's gain replicated, and larger.** +0.120 here against +0.056 in
  the pilot. The pilot's 100-query sample was easier (BM25 nDCG@10 0.698
  there against 0.666 here, and more queries with rank 1 already relevant),
  so it had less room to gain. The check half is closer to the full dev set
  (BM25 0.667 on all 809 queries).
- **Where the gain comes from:** almost entirely from queries whose first
  relevant document BM25 ranked 2–20. Queries already right at rank 1 lose a
  little (0.990 → 0.985), and queries beyond rank 20 cannot gain from
  reranking 20 candidates.
- **Still development data.** The 404 check queries were untouched by any
  model setting, but they belong to the same train split. The held-out test
  queries are kept for a final evaluation.
- **Contamination cannot be ruled out:** SciFact is public and may be in the
  model's training data.
- **Costs are client-side estimates.** The provider's bill is authoritative.

## Reproduce

```bash
venv/bin/python experiments/retrieval_s1_check/run_check.py --dry-run   # free: scripted model
venv/bin/python experiments/retrieval_s1_check/run_check.py             # paid: needs ANTHROPIC_API_KEY in .env
venv/bin/python experiments/retrieval_s1_check/analyze.py               # free: scores the recorded calls
```
