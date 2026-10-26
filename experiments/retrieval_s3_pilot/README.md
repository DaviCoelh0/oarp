# S3 pilot: a search agent against reranking, on a stratified tune sample

A paid pilot on 2026-10-25, run from commit `8d57c64` (clean tree). The
agent (`src/oarp/envs/retrieval/agent.py`, a LangGraph state graph) sees a
claim and BM25's top 20, may search the corpus again with its own queries (at
most 2 more searches of 10 results), and its candidates are then reranked by
the S1 reranker. Question: **is letting the model search again worth its
cost?**

    venv/bin/python experiments/retrieval_s3_pilot/run_pilot.py   # PAID (cap $1.60)
    venv/bin/python experiments/retrieval_s3_pilot/analyze.py     # free

Outputs: [`results/analysis.md`](results/analysis.md), `results/summary.json`,
`results/manifest.json` (aggregates, settings, hashes); per-episode records in
the gitignored `results/private/`.

## Design (fixed before the run)

- **Data:** SciFact train queries, **tune half only** (selection data). The
  check half and the test split were not used.
- **Strata** by BM25's first relevant rank:
  - all 63 tune queries beyond rank 20, where S1 scores exactly 0;
  - 100 of the 342 within rank 20, sampled by a seeded hash.
- **Overall estimate:** reweighted to tune's proportions (63 : 342), with a
  stratified bootstrap (10,000 resamples).
- **Smoke phase:** 5 further episodes first, excluded from the analysis. All
  5 were clean.
- **Conditions:**
  - **S1:** rerank BM25's top 20. Reused from the earlier S1 runs, with
    identical settings, pinned by hash.
  - **S3 + forced rerank:** the agent's candidates (20–40), reranked by S1.
  - **S3 + agent ordering:** the agent's own `RANKING:` line, scored from
    the same episodes. The agent's prompt does not depend on which ordering
    is used.
- **Model:** `claude-haiku-5-5`, effort medium, adaptive thinking
  (summarized), `max_tokens` 4096, no sampling parameters, parallel tool use
  off.
- **One episode per query.** Outputs are stochastic.
- **Harness settings:** forced first search (BM25 top 20 at 2,000
  characters), at most 3 searches in total, 10 results at 800 characters per
  model search, at most 8 agent calls, forced rerank.
- **Model behaviour:** whether and what to search, and the agent's own order.

## Results (n = 163 episodes; run valid)

nDCG@10 means:

| stratum | S0 (BM25) | S1 | S3 + forced rerank | S3 + agent ordering |
|---|---|---|---|---|
| beyond 20 (63) | 0.000 | 0.000 | 0.236 | 0.220 |
| within 20 (100) | 0.795 | 0.895 | 0.914 | 0.912 |

Pre-registered hypotheses (95% CIs):

| | Difference | Reading |
|---|---|---|
| **H1** overall (stratified), S3 + forced rerank − S1 | **+0.053 [+0.035, +0.073]** | positive |
| H2 beyond 20, same difference | +0.236 [+0.156, +0.322] | holds |
| H3 within 20, same difference (margin −0.02) | +0.019 [+0.005, +0.038] | holds, no harm |
| H4 agent ordering − forced rerank, overall | −0.005 [−0.020, +0.011] | inconclusive (also per stratum) |

The pre-registered next-step rule is met: S3 at these settings earns a
confirmation on the check half.

**Mechanism** (computed from labels, analysis only):
- A model search added a relevant document for **24 of the 63** beyond-20
  queries (38%).
- The forced rerank put all 24 into the top 10; the agent's own ordering put
  22 there.
- No within-20 query lost a relevant document that S1 had in its top 10
  (0/100).
- For context, RM3 (free, untuned) brings a relevant document into its top
  20 for 10 of these 63 queries. This is descriptive: the two methods see
  different numbers of results.

**Model behaviour:**
- The agent searched again in 89% of beyond-20 episodes and in 77% of
  within-20 episodes. That is about one search per episode in both strata
  (1.02 and 0.87), so it searches almost regardless of whether BM25 already
  found something.
- Its queries are reformulations (synonyms, spelled-out mechanisms, corrected
  terms), about 7 words long. None contains a document id.
- 162 episodes finished. One ended in a safety decline; one rerank call was
  also declined. Declines: 2 of 163 episodes. No other model errors, no
  rejected tool calls.

**Cost and latency** (estimated from usage):

| | per query (tune-weighted) |
|---|---|
| S3 + forced rerank | $0.00408, 10.3 s |
| S1 | $0.00129, 4.3 s |

- That is about 3.2× the cost and 2.4× the latency.
- The rerank call is 37.5% of S3's cost.
- About $0.047 extra per rescued query.
- The whole run: $0.716 estimated (cap $1.60). The largest single call was
  $0.0024.

## Interpretation and limitations

- **Searching again helps,** and almost all of the gain comes from queries
  where BM25 had missed every relevant document. Reranking alone cannot fix
  those. In this sample it costs no quality where BM25 had already succeeded.
- **Who ranks matters little:** the agent's own order is not
  distinguishable from the forced rerank (H4). The forced rerank is the
  simpler and slightly better choice; dropping it would save about 37% of the
  cost.
- **Selection data only:** tune half, one episode per query, and a stratum
  sample whose strata were formed from dev labels. This is not a
  confirmation; the check half (55 beyond-20 queries) is next, and the test
  split comes last.
- **The within-stratum gain (+0.019) is a mixed quantity.** Extra candidates
  can contain further relevant documents. Part of the difference is also
  run-to-run noise, because the S1 reranks come from runs 1–2 days earlier.
- **The model's decision has not been isolated.** The agent searches in most
  episodes anyway, so a fixed "always reformulate once, then rerank" pipeline
  might do about as well, more simply. That comparison was deferred until S3
  showed a gain. It now did, so the question is open.
- **Contamination:** SciFact may be in the model's training data. Handles
  stop document ids from leaking, but claims may still be recognised.
- One dataset, one model, one setting of the search budget.
