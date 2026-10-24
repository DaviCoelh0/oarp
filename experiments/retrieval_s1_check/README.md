# S1 confirmation on the development check half

**Status: not run yet.** The scripts and their tests exist; no model call has
been made for this study.

## Design (fixed before the run)

- **Question:** does the reranking gain measured in the
  [S1 pilot](../retrieval_s1_pilot/README.md) (on 100 tune queries) hold
  **with nothing changed** on the 404 queries of the check half of
  SciFact's train split? No setting has been chosen on these queries.
- **Condition:** S1-plain exactly as in the pilot: Claude Haiku 5.5 reranks
  BM25's top 20 in one listwise call, with candidates shown in BM25 order.
  The runner refuses to start if the prompts or the reranker settings differ
  from those recorded in the pilot's manifest.
- **Schedule:** all 404 queries once, then passes 2 and 3 on 20 of them
  (chosen by a seeded hash of the query id) to measure stability. 444 calls,
  sequential, standard API.
- **Primary analysis:** paired difference in nDCG@10 (S1-plain − BM25), with a
  paired bootstrap 95% CI over the 404 queries. A query whose call was
  declined or failed keeps the BM25 order. A sensitivity analysis excluding
  safety-declined queries is fixed in advance.
- **Rules fixed in advance:**
  - invalid if model errors other than safety declines exceed 5%;
  - a warning if declines exceed 10%;
  - confirmed / not confirmed / reversed by whether the CI is above, around
    or below 0;
  - a cost surprise above $0.0026 per call.
- **Budget:** estimated $0.58–0.71; a client-side guard stops before any call
  that could push the estimated spend over $1.00.
- The held-out test split is not used. A confirmed result here is still a
  development-data result.

## Reproduce

```bash
venv/bin/python experiments/retrieval_s1_check/run_check.py --dry-run   # free: scripted model
venv/bin/python experiments/retrieval_s1_check/run_check.py             # paid: needs ANTHROPIC_API_KEY in .env
venv/bin/python experiments/retrieval_s1_check/analyze.py               # free: scores the recorded calls
```

Per-call records and per-query metrics stay in a gitignored `results/private/`
folder; public results are aggregates.
