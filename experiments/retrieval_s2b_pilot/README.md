# S2b gate pilot: rerank only when a model doubts BM25's rank 1

A paid pilot on 2026-10-24, run from commit `d6c4547` (clean tree). **600 gate
calls, $0.155 estimated; no new rerank calls.** The design was fixed before
the run (below). It used 100 development queries; it is not a held-out
result.

## Design (fixed before the run)

- **Question:** reranking every query helped in the
  [S1 pilot](../retrieval_s1_pilot/README.md), but most queries already had a
  relevant first result. Does one cheap model call ("is result 1 relevant to
  the claim?") decide *when* to rerank better than a free rule on BM25's
  score gap, at the same rerank rate? And does it pay for itself?
- **Gate:** Claude Haiku 5.5 sees the claim and BM25's top 3 abstracts and
  answers `ANSWER: yes` or `ANSWER: no`. "No", an unparseable answer or an
  error means "rerank". Effort `medium` is the primary condition, `low` the
  secondary one.
- **No new rerank calls:** the same 100 queries as the S1 pilot. Gated
  rankings are assembled offline from the pilot's recorded S1-plain rankings,
  and its per-query rerank costs and latencies are reused. The pilot's files
  were checked against pinned checksums.
- **Schedule:** 3 passes × 100 queries × 2 effort levels (600 calls).
- **Primary analysis:** a decision is right when the gate reranks exactly the
  queries whose BM25 rank 1 is not relevant. The primary measure is the share
  of right decisions, model gate (medium) minus the score-gap rule at the
  same rerank rate, with a paired bootstrap 95% CI. Decisions are the primary
  measure because reranking changed nDCG@10 on only 21 of these 100 queries.
- **Rules fixed in advance:**
  - invalid if non-decline model errors exceed 5%;
  - a format problem above 10% unparseable answers;
  - "no saving" if the gate costs as much as always reranking;
  - "unstable" if decisions flip on more than 20% of queries.

## Results

**Primary: model gate (medium) − score-gap rule, share of right decisions =
+0.06, 95% CI [−0.05, +0.18] (n = 100). Not better** by the pre-stated rule.
The run is valid: no format problem, the gate saves money against always
reranking, and its decisions are stable.

| gate | rerank rate | right decisions | rule at same rate | precision* | recall* | flip rate | $ per gate call | median latency |
|---|---|---|---|---|---|---|---|---|
| model, medium effort | 34% | 80% | 74% | 0.82 | 0.87 | 2% | $0.00024 | 2.2 s |
| model, low effort | 37% | 79% | 73% | 0.83 | 0.84 | 4% | $0.00021 | 1.0 s |

\* Positive class: "rank 1 is relevant", that is, keep BM25's order.

Retrieval quality and cost per query, using the S1 pilot's rankings:

| strategy | rerank rate | nDCG@10 | − always rerank (95% CI) | share of reranking gain kept | $ per query | s per query |
|---|---|---|---|---|---|---|
| BM25 only | 0% | 0.698 | −0.056 [−0.095, −0.021] | 0% | $0 | 0 |
| always rerank (S1) | 100% | 0.754 | — | 100% | $0.00130 | 4.4 |
| perfect gate (labels) | 38% | 0.761 | +0.007 [0.000, +0.018] | 113% | $0.00049 | 1.7 |
| model gate, medium | 34% | 0.729 | −0.025 [−0.049, −0.007] | 55% | $0.00067 | 3.6 |
| score-gap rule, same rate | 34% | 0.736 | −0.018 [−0.042, +0.003] | 69% | $0.00043 | 1.5 |
| model gate, low | 37% | 0.729 | −0.025 [−0.049, −0.007] | 55% | $0.00067 | 3.0 |
| score-gap rule, same rate | 37% | 0.740 | −0.014 [−0.037, +0.005] | 75% | $0.00047 | 1.6 |

- **The model gate is right more often, but on the wrong queries.** It makes
  6 points more right decisions than the free rule (CI includes 0). Yet it
  keeps less of the reranking gain, because reranking only changes the result
  on 21 of these queries. It reranks only 9 of the 16 queries that reranking
  improves.
- **Cost:**
  - The model gate halves the cost of always reranking ($0.00067 vs $0.00130
    per query) but loses 0.025 nDCG@10. Its CI excludes 0.
  - The free rule at the same rerank rate costs less ($0.00043) and loses
    less. Its CI includes 0.
- **Low vs medium effort:**
  - Almost the same decisions: they differ on 3 of 100 queries, none of
    which matters for nDCG. Right decisions: low − medium = −0.01
    [−0.05, +0.02].
  - Low used less than half the output tokens (60 vs 139) and was twice as
    fast (median 1.0 s vs 2.2 s).
- **Declined requests:** 6 of 300 medium calls and 3 of 300 low calls, all on
  2 queries that the S1 pilot's reranker was also declined on. A declined
  gate call counts as "rerank", which on those queries changes nothing.
- **Output format:** no unparseable answers.

`results/summary.json` and `results/analysis.md` hold the aggregates.
Per-call records stay in a gitignored `results/private/` folder.

## Interpretation and limitations

- **On these queries, a model gate is not worth its call.** The free
  score-gap rule did at least as well for nDCG, at lower cost and latency.
- **A perfect gate would beat always reranking** (0.761 vs 0.754 at 38% of
  the calls), because skipping queries with a correct rank 1 also avoids the
  few cases where reranking hurts. Neither gate tested here gets close to
  that.
- **Small n, a pilot.** 100 queries, of which only 21 can change nDCG. The
  primary CI is wide. The sample is also easier than the dev set (rank 1
  relevant for 62% of queries).
- **One gate prompt,** with result 1 judged in the context of results 2 and 3.
  Other prompts or more context are untested.
- **Contamination cannot be ruled out:** SciFact is public.
- **Costs are client-side estimates.** Per-query rerank costs come from the
  S1 pilot. For its declined calls, the pilot's mean rerank cost is used.

## Reproduce

```bash
venv/bin/python experiments/retrieval_s2b_pilot/run_gate.py --dry-run   # free: scripted model
venv/bin/python experiments/retrieval_s2b_pilot/run_gate.py             # paid: needs ANTHROPIC_API_KEY in .env
venv/bin/python experiments/retrieval_s2b_pilot/analyze.py              # free: scores the recorded calls
```

Needs the S1 pilot's local results, including its gitignored private records.
