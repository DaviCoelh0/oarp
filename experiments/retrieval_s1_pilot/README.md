# S1 pilot: model reranking of BM25's top 20 (plain vs shuffled presentation)

A paid pilot on 2026-10-24, run from commit `08f31b4` (clean tree, no
untracked files). **280 calls, $0.383 estimated.** The study design was fixed
before the run (below). This is a pilot on 100 development queries, not a
held-out evaluation.

## Design (fixed before the run)

- **Question:** does Claude Haiku 5.5, reranking BM25's top 20 in one listwise
  call per query, improve nDCG@10 over BM25? And does the order in which
  candidates are shown matter?
- **Conditions,** on the same queries:
  - **S0:** BM25 order.
  - **S1-plain** (primary): candidates shown in BM25 order.
  - **S1-shuffled** (secondary): candidates shown in a per-query permutation
    seeded by the query id; the answer is mapped back before scoring.
- **Queries:** 100 development queries from the tune half of SciFact's train
  split, chosen by a seeded hash of the query id (no labels). The held-out
  test split was not used.
- **Schedule:** both conditions per query back to back, in a hash-determined
  order. Two extra passes on 20 of the queries measure stability. 280 calls
  in total, sequential, standard API.
- **Model:** `claude-haiku-5-5`, effort `medium`, adaptive thinking
  (summarized), `max_tokens` 4096, no sampling parameters, no tools. The
  prompt asks for one line, `RANKING:` followed by the abstract numbers;
  prompt hashes are in `results/manifest.json`.
- **Primary analysis:** paired difference in nDCG@10 (S1-plain − S0), with a
  paired bootstrap 95% CI over queries (10,000 resamples, fixed seed).
  Secondary: S1-shuffled − S1-plain.
- **Rules fixed in advance:**
  - invalid if model errors exceed 5% in a condition;
  - a format problem if more than 10% of answers needed repair;
  - a cost surprise if a call costs more than $0.0036 on average.
  
  Spending had a client-side cap of $0.80.

## Results

**Primary (H1), S1-plain − S0: +0.056 nDCG@10, 95% CI [+0.021, +0.095]
(n = 100).** Positive by the pre-stated rule (CI above 0).

| | S0 (BM25) | S1-plain | S1-shuffled | perfect top-20 reranker |
|---|---|---|---|---|
| nDCG@10 | 0.698 | 0.754 | 0.750 | 0.830 |
| Recall@10 | | 0.820 | 0.820 | |
| MRR@10 | | 0.730 | 0.724 | |
| share of the possible gain | | 42% | 39% | 100% |

- **Secondary (H2), S1-shuffled − S1-plain: −0.004 [−0.013, +0.002].**
  Inconclusive by the pre-stated rule. The CI is narrow and close to 0, so
  presentation order made little difference to quality here. This does not
  show the two are equivalent. S1-shuffled − S0: +0.052 [+0.014, +0.093].
- **Where the change comes from** (S1-plain, the 96 queries answered): 16
  queries improved, 5 got worse, 75 were unchanged in nDCG@10. Harm rate
  was 5% overall, and 3% where BM25's rank 1 was already relevant.
- **Presentation order.**
  - No answer simply copied the shown order. Only 3% (plain) and 4%
    (shuffled) of final top-20 lists equal BM25's.
  - The correlation between where a candidate was shown and where the model
    ranked it is 0.44 with BM25 order and 0.06 when shuffled. With BM25
    order shown, the model's ranking leans on it, but quality was about the
    same either way. Shuffled presentation shows little position bias.
- **Output format:** every answered call returned a valid full permutation
  (0 repairs, 0 truncations).
- **Declined requests:** 7 of 280 calls (plain 3/140 = 2.1%, shuffled 4/140 =
  2.9%), on 4 of the 100 queries, were declined by the provider's safety
  classifier (category "bio"). They were not retried. As specified, they
  count as model errors, and the query keeps the BM25 order the harness
  falls back to. This is below the 5% invalid threshold.
  - Exploratory check, not specified in advance and run as a one-off
    analysis outside the committed scripts: leaving out the 4 declined
    queries gives S1-plain − S0 = +0.059 [+0.022, +0.099], and S1-shuffled −
    S0 = +0.054 [+0.015, +0.096] (n = 96).
- **Stability** (20 queries × 3 passes per condition): nDCG@10 was identical
  across passes for all 20 queries in both conditions. The orderings differ
  (mean Kendall τ between passes on the top 20: 0.78 plain, 0.76 shuffled;
  top-1 agreement 95%), but the relevant documents landed in the same
  positions.

### Cost and latency (measured)

| | S1-plain | S1-shuffled |
|---|---|---|
| input tokens, mean / p90 | 10,860 / 12,628 | 10,866 / 12,628 |
| output tokens, mean / p90 (incl. thinking) | 462 / 584 | 446 / 578 |
| estimated $ per call | $0.00132 | $0.00131 |
| latency per call, median / p90 | 4.4 s / 5.6 s | 4.2 s / 5.6 s |

- Total estimated spend: $0.383 for 280 calls (cap $0.80; the guard never
  triggered).
- The cost per call fell within the pre-run estimate of $0.0009–0.0018.
  However, input tokens were about 44% above the estimate: the estimate
  assumed about 4 characters per token, and SciFact abstracts come out
  nearer 2.8.
- For comparison, BM25 alone costs nothing and scores all 809 dev queries
  in about 2 s.

Aggregates: `results/summary.json`, `results/analysis.md`. Per-call records
(answers, thinking summaries, rankings) and per-query metrics are derived
from the labels or tied to query ids, so they stay in a gitignored
`results/private/` folder.

## Interpretation and limitations

- **Reranking helps on these queries, at a cost.** About +0.056 nDCG@10 for
  about $0.0013 and 4–5 s per query, recovering about 40% of what a perfect
  reranker of the same 20 candidates could gain. Most queries are unchanged;
  the gain comes from about one in six.
- **The sample is somewhat easier than the tune half.** BM25 has rank 1
  relevant for 62% of the sample against 57% of the tune half (S0 nDCG@10
  0.698 against 0.668 on the tune half). The sample was fixed before the
  run without labels and was not changed.
- **n = 100, one pass per query** for the primary analysis. The CI excludes
  0 but is wide relative to the effect (+0.02 to +0.10).
- **Tune queries only.** The prompt and settings may still be revised on
  tune, so a confirmatory claim needs the check half. This is not a held-out
  result.
- **Contamination cannot be ruled out:** SciFact is public and may be in the
  model's training data. That would inflate both S1 conditions.
- **Safety declines are a real cost of this setup:** about 2.5% of calls on
  biomedical claims returned no ranking.
- **Costs are client-side estimates** from a dated price table. The
  provider's bill is authoritative.

## Reproduce

```bash
venv/bin/python experiments/retrieval_s1_pilot/run_pilot.py --dry-run   # free: scripted model
venv/bin/python experiments/retrieval_s1_pilot/run_pilot.py             # paid: needs ANTHROPIC_API_KEY in .env
venv/bin/python experiments/retrieval_s1_pilot/analyze.py               # free: scores the recorded calls
```

The paid run refuses to start unless:
- the code is committed;
- `results/` is empty;
- the dataset matches its pinned checksum.
