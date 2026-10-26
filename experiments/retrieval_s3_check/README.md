# S3 confirmation on the development check half

A paid study on 2026-10-26, run from commit `a6ed70b` (clean tree). It
confirms the [S3 pilot](../experiments/retrieval_s3_pilot/README.md) on
queries that played no part in designing S3. The search agent (S3) sees a
claim and BM25's top 20, may search again (at most 2 more searches of 10),
and its candidates are reranked by the S1 reranker. The comparison is
against S1 (reranking BM25's top 20 only).

    venv/bin/python experiments/retrieval_s3_check/run_check.py   # PAID (cap $2.20)
    venv/bin/python experiments/retrieval_s3_check/analyze.py     # free

Outputs: [`results/analysis.md`](results/analysis.md), `results/summary.json`,
`results/manifest.json` (aggregates, settings, hashes); per-episode records in
the gitignored `results/private/`.

## Design (fixed before the run)

- **All 404 check queries** of SciFact's train split: 55 with no relevant
  document in BM25's top 20, 349 with one. The test split was not loaded.
- **Blind for S3:** S3 had never run on a check query.
- **Settings frozen at the pilot's:** the same config and prompt hashes,
  `claude-haiku-5-5`, effort medium, adaptive thinking, `max_tokens` 4096,
  no sampling parameters. One episode per query.
- **S1 reused:** the S1 check study's records (2026-10-24), pinned by hash.
- **Smoke phase:** 5 episodes on spare tune queries, excluded from the
  analysis; all 5 were clean.
- **Pre-registered rule (95% paired bootstrap CIs, 10,000 resamples):**
  - **confirmed** if C1 (S3 + forced rerank − S1 > 0 overall) and C2 (on the
    349 within-20 queries, no loss beyond −0.02) both hold;
  - **confirmed with harm** if only C1 holds;
  - otherwise **not confirmed**.

## Results (run valid)

nDCG@10 means:

| queries | S0 (BM25) | S1 | S3 + forced rerank | S3 + agent ordering |
|---|---|---|---|---|
| all 404 | 0.666 | 0.786 | **0.819** | 0.824 |
| beyond 20 (55) | 0.000 | 0.000 | 0.312 | 0.296 |
| within 20 (349) | 0.771 | 0.910 | 0.899 | 0.908 |

| | Difference [95% CI] | Reading |
|---|---|---|
| **C1** all queries, S3 + forced rerank − S1 | **+0.033 [+0.014, +0.053]** | holds |
| **C2** within 20, same difference (margin −0.02) | −0.011 [−0.021, −0.002] | **fails**: the lower bound −0.0211 is below −0.02 |
| C3 beyond 20, same difference | +0.312 [+0.211, +0.422] | holds |
| C4 agent ordering − forced rerank (descriptive) | overall +0.005 [−0.004, +0.016] | inconclusive (also per stratum) |
| S3 + forced rerank − S0 | +0.153 [+0.125, +0.181] | |

**Pre-registered outcome: confirmed with harm.** S3 + forced rerank
improves on S1 over the check half. On the queries where BM25 already had a
relevant document in its top 20, it is slightly worse than S1, and the
interval reaches just past the pre-set margin.

**Mechanism** (computed from labels, analysis only):
- A model search added a relevant document for 24 of the 55 beyond-20
  queries (44%). The forced rerank put all 24 into the top 10.
- Within 20: 306 of 349 queries scored the same as S1, 17 better and 26
  worse. In 4 of them, a relevant document from S1's top 10 was lost.
- The loss sits in the 269 within-20 episodes where the agent added
  documents (mean −0.015). The 80 without new documents average +0.002. More
  candidates (up to 40) make the reranker's task harder; it was confirmed
  only at 20.
- The beyond-20 queries carry 129% of the overall gain; the within-20
  queries take back 29%.

**Model behaviour:**
- The agent searched again in 93% of beyond-20 and 83% of within-20
  episodes, about one search per episode in both.
- 370 model queries; none contains a document id.
- 398 episodes finished. Declines by the provider's safety classifier hit 9
  episodes (2.2%): 5 agent calls and 9 rerank calls. There were no other
  model errors and no rejected tool calls.

**Cost and latency** (estimated from usage):

| | per query |
|---|---|
| S3 + forced rerank | $0.00411, 11.2 s |
| S1 | $0.00129, 4.2 s |

- About 3.2× the cost and 2.7× the latency.
- About $0.047 extra per rescued query.
- The whole run: $1.757 estimated (cap $2.20). The largest call was $0.0025.

**Tune → check:** the pilot, reweighted to check proportions, gave +0.049
[+0.032, +0.068]; check gives +0.033 [+0.014, +0.053]. The effect shrank by
about a third but stays clearly positive.

## Interpretation and limitations

- **Searching again is confirmed to help overall,** through the queries
  BM25 missed entirely. The within-20 harm is small (−0.011), but by the
  pre-set rule it is not excluded beyond −0.02.
- **Whether the agent should order instead of the reranker is not
  answered.** On the within-20 queries the agent's own ordering came out
  almost level with S1 (0.908 against 0.910). But C4 is inconclusive, and
  choosing agent ordering because of this check result would be selection
  on check data. That choice would need new tune evidence and a test-split
  evaluation.
- **Candidates for the next design step** (each needs new tune evidence;
  none can be confirmed on this check half again):
  - rerank only the original 20 plus the documents the model's searches
    added, with fewer slots;
  - rerank at most 20 candidates;
  - let the agent's order decide within the original top 20.
- **The model's decision is still not isolated.** The agent searches in
  most episodes anyway; a fixed "reformulate once, then rerank" pipeline is
  the approved next comparison (a tune pilot).
- **Further caveats:** one episode per query (stochastic); the S1 reranks
  come from an earlier run, so run-to-run noise lands in the difference;
  SciFact may be in the model's training data; one dataset, one model, one
  search budget.
- **This is development data.** The final, held-out evaluation on the test
  split is still to come.
