# Automated configuration search over retrieval policies (offline, $0)

This is a bounded search over **harness policies** that decide, per query,
how much retrieval work to spend. The model is frozen; only the harness
routing changes. Each policy uses BM25's own scores (no labels, no model
call) to send a query to one of three recorded pipelines:

    if search_signal < τ_search:   S3  (search agent + forced rerank; about $0.0041)
    elif rerank_signal < τ_rerank: S1  (rerank BM25's top 20; about $0.0013)
    else:                          S0  (BM25's order; $0)

    venv/bin/python experiments/retrieval_config_search/search.py   # free: no model calls

Outputs: [`results/analysis.md`](results/analysis.md) (including the Pareto
frontier), `results/summary.json`; per-query details in the gitignored
`results/private/`.

## Space and rule (fixed before any score was computed)

- **Signals,** for each decision: the relative gap between BM25's top two
  scores, the raw gap, and the top score.
- **Rates:** 0, 0.1, …, 1.0 for each decision. τ is the signal's quantile
  over all 809 development queries, so it is label-free.
- **Size:** 3 × 11 × 3 × 11 = **1,089 policies**, of which 591 have distinct
  outcomes.
- **Scoring:** every policy is scored offline on the **404 check queries**,
  where S0, S1 and S3 are all recorded (from the S1 check study and the S3
  check study). Costs are the recorded per-query estimates.
- **Selection rule:** the **cheapest policy whose mean nDCG@10 is within
  0.005 of the best policy's**.

## Result (development data; the selected numbers are optimistic)

| policy | nDCG@10 (check) | $ per query | s per query | routed to S3 / S1 / S0 |
|---|---|---|---|---|
| S0 (BM25) | 0.666 | 0 | 0 | 0 / 0 / 100% |
| S1 (always rerank) | 0.786 | 0.00131 | 4.2 | 0 / 100% / 0 |
| S3 (always search) | 0.819 | 0.00411 | 11.2 | 100% / 0 / 0 |
| best policy | 0.821 | 0.00325 | 9.1 | 70% / 24% / 6% |
| **selected policy** | **0.816** | **0.00196** | **5.8** | **29% / 53% / 18%** |

- **The selected policy:** search if BM25's top score < 9.8509 (the lowest
  30% of development queries); else rerank if the raw gap < 7.0108 (rate
  0.8); else keep BM25's order.
- **On check:**

  | | Difference [95% CI] |
  |---|---|
  | selected − S1 | +0.030 [+0.013, +0.048] |
  | selected − S3 | −0.003 [−0.013, +0.006] |
  | selected − S0 | +0.150 [+0.123, +0.178] |

  That is close to the agent's quality at about **half its cost and half
  its latency**.
- **Tune cross-check** (stratified, descriptive): 0.790 at $0.00211, against
  S1 at 0.756 and S3 at 0.809. The policy's edge over S1 holds there; its
  closeness to S3 is weaker, as expected for a policy selected on check.

## Limitations

- **These are selection numbers.** The policy was chosen on these 404
  queries, so its check scores are optimistic. Only the untouched test split
  can validate it.
- **One recorded episode per query,** with records from different days.
  The routes reuse recorded outcomes, which assumes a pipeline's outcome
  does not depend on which other queries were routed to it (true here: each
  query is independent).
- **A small, fixed space** of three signals and two thresholds. This is
  configuration search, not a learned or self-improving system.
