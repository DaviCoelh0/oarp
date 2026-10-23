# S1 pilot: model reranking of BM25's top 20 (plain vs shuffled presentation)

**Status: not run yet.** The scripts and their tests exist; no model call has
been made. This README will hold the results once the paid run is done.

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
  test split is not used.
- **Schedule:** both conditions per query back to back, in a hash-determined
  order. Two extra passes on 20 of the queries measure stability. 280 calls
  in total.
- **Model:** `claude-haiku-5-5`, effort `medium`, adaptive thinking (summarized),
  `max_tokens` 4096, no sampling parameters, no tools. Calls are sequential on
  the standard API.
- **Primary analysis:** paired difference in nDCG@10 (S1-plain − S0) with a
  paired bootstrap 95% CI over queries. Secondary: S1-shuffled − S1-plain.
  Also reported: quality next to calls, tokens, estimated cost and latency;
  output validity; presentation diagnostics; stability across repeats.
- **Budget:** estimated $0.25–0.50. A client-side guard stops before any call
  that could push the estimated spend over $0.80.

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

Per-call records, rankings and per-query metrics stay in a gitignored
`results/private/` folder; public results are aggregates.
