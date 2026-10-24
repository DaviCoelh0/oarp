# S2b gate pilot: rerank only when a model doubts BM25's rank 1

**Status: not run yet.** The scripts and their tests exist; no model call has
been made for this study.

## Design (fixed before the run)

- **Question:** reranking every query with a model helped in the
  [S1 pilot](../retrieval_s1_pilot/README.md), but most queries already had a
  relevant first result. Does one cheap model call ("is result 1 relevant to
  the claim?") decide *when* to rerank better than a free rule on BM25's
  score gap, at the same rerank rate? And does it pay for itself?
- **Gate:** Claude Haiku 5.5 sees the claim and BM25's top 3 abstracts and
  answers `ANSWER: yes` or `ANSWER: no`. "No", an unparseable answer or an
  error means "rerank". Effort `medium` is the primary condition, `low` the
  secondary one.
- **No new rerank calls:** the same 100 development queries as the S1 pilot.
  Gated rankings are assembled offline from the pilot's recorded rankings,
  and its per-query rerank costs are reused. The runner refuses to start if
  the pilot's files differ from their pinned checksums.
- **Schedule:** 3 passes × 100 queries × 2 effort levels (600 calls), both
  efforts back to back per query, in a hash-determined order.
- **Primary analysis:** a decision is right when the gate reranks exactly the
  queries whose BM25 rank 1 is not relevant. The primary measure is the share
  of right decisions, model gate (medium) minus the score-gap rule at the
  same rerank rate, with a paired bootstrap 95% CI. Decisions are the primary
  measure because reranking changed nDCG@10 on only 21 of these 100 queries,
  which leaves an nDCG comparison between gates with little power.
- **Also reported:**
  - nDCG@10 and the share of the reranking gain kept;
  - cost per query (gate call + rerank where reranked) against always
    reranking;
  - latency;
  - precision and recall of the gate;
  - how often its decision changes across passes;
  - low vs medium effort.
- **Rules fixed in advance:**
  - invalid if non-decline model errors exceed 5%;
  - a format problem above 10% unparseable answers;
  - "no saving" if the gate costs as much as always reranking;
  - "unstable" if decisions flip on more than 20% of queries.
- **Budget:** estimated $0.12–0.25; a client-side guard stops before any call
  that could push the estimated spend over $0.50.

## Reproduce

```bash
venv/bin/python experiments/retrieval_s2b_pilot/run_gate.py --dry-run   # free: scripted model
venv/bin/python experiments/retrieval_s2b_pilot/run_gate.py             # paid: needs ANTHROPIC_API_KEY in .env
venv/bin/python experiments/retrieval_s2b_pilot/analyze.py              # free: scores the recorded calls
```

Needs the S1 pilot's local results, including its gitignored private records.
Per-call records stay in a gitignored `results/private/` folder; public results
are aggregates.
