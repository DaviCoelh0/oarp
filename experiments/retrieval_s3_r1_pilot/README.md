# R1 pilot: "reformulate once, then rerank" against the search agent

A paid pilot on 2026-10-26, run from commit `2b8c12b` (clean tree).

**Question:** does the agent's decision (whether, how often and what to
search, after seeing results) add anything over a fixed harness rule that
always makes exactly one model-written search?

**What R1 is:**
- The model sees exactly the agent's first message (the claim plus BM25's
  top 20, byte for byte). Only the closing instruction differs: it must
  reply with one `QUERY:` line.
- The harness runs that one search (10 results), and the S1 reranker orders
  all candidates.
- There are no tools and no second search.
- Graph: `forced_search → reformulate → search → rerank → assemble`.

    venv/bin/python experiments/retrieval_s3_r1_pilot/run_pilot.py   # PAID (cap $0.80)
    venv/bin/python experiments/retrieval_s3_r1_pilot/analyze.py     # free

Outputs: [`results/analysis.md`](results/analysis.md), `results/summary.json`,
`results/manifest.json`; per-episode records in the gitignored
`results/private/`.

## Design (fixed before the run)

- **Queries:** exactly the [S3 pilot](../retrieval_s3_pilot/README.md)'s 163
  tune queries: all 63 whose first relevant document is beyond BM25 rank 20,
  plus 100 of the 342 within.
- **Pairing:** each R1 episode is paired with the S3 pilot episode (S3 +
  forced rerank, 2026-10-25) and the reused S1 record for the same query.
- **Smoke phase:** 5 smoke episodes on unused tune queries first; all were
  clean.
- **Settings:** identical to S3 apart from `extra_search="once"`.
  `claude-haiku-5-5`, effort medium, adaptive thinking, `max_tokens` 4096.
  One episode per query.
- **Primary comparison:** R1 − S3 in nDCG@10, stratified with tune weights
  (63 : 342), 95% bootstrap CI, two-sided. Readings:
  - "S3's decision adds value": the CI lies below 0;
  - "R1 better": the CI lies above 0;
  - "equivalent within ±0.02": the CI lies inside [−0.02, +0.02];
  - otherwise inconclusive.

## Results (n = 163; run valid)

nDCG@10 means:

| stratum | S0 | S1 | S3 (agent) | R1 (fixed) |
|---|---|---|---|---|
| beyond 20 (63) | 0.000 | 0.000 | 0.236 | 0.289 |
| within 20 (100) | 0.795 | 0.895 | 0.914 | 0.887 |

| | Difference [95% CI] | |
|---|---|---|
| **R1 − S3, overall (primary)** | **−0.015 [−0.038, +0.007]** | **inconclusive** |
| R1 − S3, beyond 20 | +0.054 [−0.018, +0.133] | |
| R1 − S3, within 20 | −0.027 [−0.051, −0.005] | |
| R1 − S1, overall | +0.038 [+0.018, +0.059] | |
| R1 − S1, within 20 (margin −0.02) | −0.008 [−0.025, +0.009] | harm beyond the margin not excluded |

**Mechanism** (computed from labels, analysis only):
- **Rescues:** R1 added a relevant document for 26 of the 63 beyond-20
  queries, S3 for 24.
- **Overlap:** only 18 queries were rescued by both, 32 by either. The
  documents the two searches added overlap little (mean Jaccard 0.21).
- **Search volume:** R1 adds more documents per episode (beyond 6.3, within
  4.4) than S3 (5.6, 3.5). S3 skipped searching in 11% of beyond-20 and 23%
  of within-20 episodes.
- **Lost documents:** within 20, R1 lost a relevant document from S1's top
  10 in 1 of 100 queries; S3 in none.

**Behaviour and errors:**
- 161 valid reformulations, about 10 words each, none containing a document
  id.
- 2 episodes were declined by the provider's safety classifier (the
  reformulation call and the rerank call). There were no other errors.

**Cost and latency** (estimated, tune-weighted per query):

| | $ per query | s per query |
|---|---|---|
| R1 | 0.00267 | 7.5 |
| S3 | 0.00408 | 10.3 |
| S1 | 0.00129 | 4.3 |

The whole run cost $0.472 estimated (cap $0.80); the largest call was
$0.0022.

## Interpretation and limitations

- **The primary question is not settled.** At this sample size the fixed
  rule is neither shown equivalent to the agent nor shown different.
- **The two parts point in opposite directions.** These are secondary
  results, not tested hypotheses.
  - **Queries BM25 missed:** R1 does at least as well as S3 (+0.054, CI
    includes 0).
  - **Queries BM25 already got right:** R1 is worse than S3 (−0.027, CI
    excludes 0).
  - This fits the check-half finding that harm comes with added candidates:
    R1 always searches, while the agent sometimes does not.
- **R1 is 35% cheaper and 27% faster than S3** per query, and also beats S1
  on tune (+0.038).
- **Different queries find different documents.** Only 18 of the 32
  rescued queries were rescued by both. This suggests the gain depends on
  which query is written, and that it varies from run to run.
- **Caveats:**
  - tune data only (selection data), one episode per query;
  - the S3 episodes come from a run one day earlier, so run-to-run noise
    adds to the paired differences;
  - one dataset, one model;
  - SciFact may be in the model's training data.
- **Not established:** that the agent's decision adds value overall, or
  that R1 can replace it. The within-20 pattern suggests a further design to
  study on tune: search only when BM25 looks unsure (the S2 score-gap
  signal), or cap the reranked candidates. Either would need its own
  approval.
