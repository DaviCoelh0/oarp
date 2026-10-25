# S2 threshold: choose on tune, confirm on check (not blind)

A paid study on 2026-10-24, run from commit `f216160` (clean tree, no
untracked files). **305 rerank calls: $0.392 for the calls that returned
usage, $0.412 counted by the budget guard** (declined calls at the $0.004
worst case). The design was fixed before the run (below). Development data
only.

**Result: not shown.** The threshold chosen on tune saved half the rerank
calls there at a small loss, but on check it lost more than the pre-stated
margin allows.

## Design (fixed before the run)

- **Question:** can a free rule on BM25's score gap (rerank only when
  (s1 − s2)/s1 < τ) keep almost all of always-reranking's quality with
  clearly fewer rerank calls?
- **Paid part:** model reranking, exactly as in the
  [S1 pilot](../retrieval_s1_pilot/README.md) (Claude Haiku 5.5, effort
  `medium`, BM25 top 20; prompts and settings checked against the pilot's
  manifest). It ran on the 305 development "tune" queries the pilot did not
  cover, one call each. With the pilot's 100, all 405 tune queries have
  reranked results.
- **Choosing τ on tune:** the threshold that reranks the fewest queries while
  staying within **0.01** nDCG@10 of reranking every query.
- **Confirming on check, offline:** τ applied unchanged to the 404 "check"
  queries, using the [check study](../retrieval_s1_check/README.md)'s
  reranked results. The rule passes if the paired bootstrap 95% CI of
  nDCG@10 (rule − always rerank) has its lower bound above **−0.02**.
- **This confirmation is not blind.** The rule's quality curve on the check
  half was computed and seen before this design was fixed. τ was chosen on
  tune only, but the tolerance and margin were set with that curve known.

## Results

**Selection on tune (405 queries):** τ* = 0.180. This reranks 50.9% of tune
queries:

| | BM25 | rule (τ*) | always rerank |
|---|---|---|---|
| tune nDCG@10 | 0.668 | 0.746 | 0.754 |

Rule − always rerank on tune: −0.008, 95% CI [−0.015, −0.002], within the
0.01 tolerance by construction.

**Confirmation on check (404 queries, not blind):**

| | BM25 | rule (τ*) | always rerank | perfect gate |
|---|---|---|---|---|
| nDCG@10 | 0.666 | 0.760 | 0.786 | 0.788 |
| rerank rate | 0% | 49.5% | 100% | 44.8% |
| $ per query | $0 | $0.00066 | $0.00131 | |
| s per query | 0 | 2.2 | 4.2 | |

- **Rule − always rerank: −0.026, 95% CI [−0.039, −0.015].** The lower bound
  is below −0.02, so the pre-stated rule says **not shown**. Excluding the 8
  queries whose rerank call was declined: −0.027 [−0.040, −0.015].
- **Rule − BM25: +0.094 [+0.072, +0.116].** The rule keeps 78% of the
  reranking gain for half the calls, cost and latency. That doesn't meet the
  bar set in advance.
- **Tune-to-check drop.** On tune the loss falls steeply around this rerank
  rate (−0.024 at 40%, −0.014 at 50%, −0.005 at 55%, −0.003 at 65%). The
  selected threshold sits on that steep part, where small differences
  between halves matter most. The check half also gains more from reranking
  (+0.120 against +0.086 on tune), so skipping reranks costs more there.
- **The 305 new tune calls** (descriptive):
  - Reranking gained +0.096 nDCG@10 on these queries: 83 improved, 16 worse,
    206 unchanged.
  - 5 calls (1.6%) were declined by the provider's safety classifier.
  - No other errors, 0 repairs.
  - About 10,750 input / 470 output tokens per call; at most $0.0017 per
    call.

`results/summary.json` and `results/analysis.md` hold the aggregates, also
showing τ matched on tune to fixed rates of 50%, 60% and 80% (on check: −0.026,
−0.012 and −0.001 against always reranking). Per-call records and per-query
values stay in a gitignored `results/private/` folder.

## Interpretation and limitations

- **A negative result for this rule at this tolerance.** Choosing the fewest
  reranks within 0.01 on tune lands where the trade-off is steep, and the
  loss on check was about three times larger (−0.026 against −0.008). A more
  conservative choice, such as a higher rerank rate, would have lost less on
  check. The fixed-rate rows suggest that. But that is hindsight on a
  non-blind half, and is not claimed as a result.
- **Not blind:** the check curve was known in advance. Here the outcome went
  against the rule anyway.
- **Development data only.** The held-out test split stays unused for a final
  evaluation.
- **Tune reranked results come from two runs of the same configuration** a
  day apart (the S1 pilot and this study). The settings and prompts were
  checked to be identical.
- **Costs are client-side estimates.** Per-query costs on check come from the
  check study's records. Declined calls are counted at that study's mean.

## Reproduce

```bash
venv/bin/python experiments/retrieval_s2_tau/run_tune.py --dry-run   # free: scripted model
venv/bin/python experiments/retrieval_s2_tau/run_tune.py             # paid: needs ANTHROPIC_API_KEY in .env
venv/bin/python experiments/retrieval_s2_tau/analyze.py              # free: selection and confirmation
```

Needs the S1 pilot's and the check study's local results, including their
gitignored private records (checked against pinned checksums).
