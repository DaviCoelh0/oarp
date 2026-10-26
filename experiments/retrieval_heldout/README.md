# Held-out evaluation on SciFact's test split

This is the final, one-shot evaluation of the retrieval line. It ran on
2026-10-26 from commit `106f9e7` (clean tree), on the **300 test queries,
which were loaded for the first time in this run**. Everything evaluated here
was designed, selected and confirmed on development data, and frozen before
the test split was opened. Nothing was changed afterwards.

    venv/bin/python experiments/retrieval_heldout/run_test.py   # PAID (cap $2.30); loads the test split
    venv/bin/python experiments/retrieval_heldout/analyze.py    # free

Outputs: [`results/analysis.md`](results/analysis.md), `results/summary.json`,
`results/manifest.json`; per-query details in the gitignored
`results/private/`.

## Conditions (all frozen in advance)

| | What it does |
|---|---|
| **S0** | BM25 (`bm25s`, k1 1.5, b 0.75) |
| **S1** | A model reranks BM25's top 20 (one call; [S1](../retrieval_s1_check/README.md)) |
| **S3** | A search agent: sees BM25's top 20, may search the corpus again (at most 2 more searches), then its candidates are reranked ([S3](../retrieval_s3_check/README.md)) |
| **P\*** | The policy chosen by the [automated configuration search](../retrieval_config_search/README.md) on development data: search (S3) if BM25's top score < 9.8509, else rerank (S1) if BM25's raw gap < 7.0108, else BM25 (S0) |

- **Model:** `claude-haiku-5-5`, effort medium, adaptive thinking,
  `max_tokens` 4096, no sampling parameters. One S1 call and one S3 episode
  per test query.
- **P\*:** scored offline from them by its routing.
- **Smoke phase:** 5 smoke queries on unused development queries first, all
  clean.

## Results (n = 300; run valid)

| | S0 | S1 | S3 | P\* |
|---|---|---|---|---|
| **nDCG@10** | 0.662 | 0.757 | **0.791** | 0.786 |
| $ per query (estimated) | 0 | 0.00132 | 0.00420 | **0.00204** |
| s per query | 0 | 4.5 | 11.0 | 5.9 |

**Pre-registered comparisons** (95% paired bootstrap CIs, 10,000 resamples):

| | Difference | Outcome |
|---|---|---|
| H1 S1 − S0 | **+0.095 [+0.067, +0.125]** | holds |
| H2 S3 − S1 | **+0.034 [+0.016, +0.055]** | holds |
| H3 P\* − S1 (primary) | **+0.030 [+0.014, +0.048]** | holds |
| H4 P\* − S3 ≥ −0.01, at a lower cost | −0.004 [−0.015, +0.004]; cost 49% of S3's | **not shown** (the lower bound −0.015 is below −0.01) |

**The optimisation claim (H3 and H4) is not established.** The searched
policy beats reranking on held-out data, and its point estimate is 0.004
below the agent at 49% of its cost and 54% of its latency. But the interval
does not exclude a loss larger than the −0.01 margin fixed in advance.

**Descriptive** (strata from test labels after the run; not tested):

| | n | S1 | S3 | P\* |
|---|---|---|---|---|
| BM25 missed every relevant document in its top 20 | 48 | 0.000 | 0.211 | 0.172 |
| BM25 had one in its top 20 | 252 | 0.901 | 0.901 | 0.903 |

- **Missed queries:** P\* gives up part of S3's gain (−0.038 [−0.090,
  +0.000]). Its BM25-confidence routing sends only some of them to the
  agent.
- **Queries BM25 already handled:** S3 shows no harm on test (+0.001
  [−0.011, +0.012]). The small loss seen on the check half did not
  reappear.
- **Routing:** P\* sent 94 queries to the agent, 156 to the reranker and 50
  to BM25 only.
- **The dev Pareto frontier evaluated on test:** each policy kept the
  cost/quality ordering it had on development data, with nDCG@10 about
  0.01–0.03 lower on test.
- **Behaviour and errors:**
  - the agent searched again in 88% of episodes;
  - 291 model queries; none contains a document id (one six-digit number
    was copied from the claim text);
  - 6 queries (2%) had a safety decline; there were no other errors;
  - the run cost $1.755 estimated (cap $2.30); the largest call was
    $0.0025.

## Interpretation and limitations

- **Held-out evidence that each step of retrieval work pays off, on
  untouched queries:**
  - model reranking improves on BM25 (+0.095);
  - an agent that may search again improves on reranking (+0.034);
  - a cost-aware routing policy found by automated search also improves on
    reranking (+0.030) at half the agent's cost.
- **Not shown:** that the routed policy matches the agent within 0.01. That
  part of the optimisation result is inconclusive, not negative.
- **Limits:**
  - one run of a stochastic model, one episode per query;
  - one dataset and one model; SciFact may be in the model's training data;
  - costs are client-side estimates from token counts.
- **The test split is now used.** Any further claim needs a new held-out
  set or another benchmark.
