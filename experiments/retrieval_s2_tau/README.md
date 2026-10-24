# S2 threshold: choose on tune, confirm on check (not blind)

**Status: not run yet.** The scripts and their tests exist; no model call has
been made for this study.

## Design (fixed before the run)

- **Question:** can a free rule on BM25's score gap (rerank only when
  (s1 − s2)/s1 < τ) keep almost all of always-reranking's quality with
  clearly fewer rerank calls?
- **Paid part:** model reranking (exactly as in the
  [S1 pilot](../retrieval_s1_pilot/README.md); the runner checks the prompts
  and settings against the pilot's manifest) on the 305 development "tune"
  queries the pilot did not cover. One call each, cap $0.70. With the pilot's
  100, all 405 tune queries then have reranked results.
- **Choosing τ on tune:** the threshold that reranks the fewest queries while
  staying within 0.01 nDCG@10 of reranking every query. If only "always
  rerank" qualifies, the rule cannot save calls.
- **Confirming on check, offline:** τ is applied unchanged to the 404 "check"
  queries, using the reranked results from the
  [check study](../retrieval_s1_check/README.md). The rule passes if the
  paired bootstrap 95% CI of nDCG@10 (rule − always rerank) has its lower
  bound above −0.02.
- **This confirmation is not blind.** The rule's quality curve on the check
  half was already computed and seen (in the check study's analysis) before
  this design was fixed. τ is chosen on tune only, but the tolerance and
  margin were set with that curve known. A blind confirmation is planned on
  the held-out test split at the final evaluation.
- **Also reported:** rerank rate and calls saved, cost and latency per query,
  the share of the reranking gain kept, a perfect gate for comparison, and
  the same with safety-declined queries excluded.

## Reproduce

```bash
venv/bin/python experiments/retrieval_s2_tau/run_tune.py --dry-run   # free: scripted model
venv/bin/python experiments/retrieval_s2_tau/run_tune.py             # paid: needs ANTHROPIC_API_KEY in .env
venv/bin/python experiments/retrieval_s2_tau/analyze.py              # free: selection and confirmation
```

Needs the S1 pilot's and the check study's local results, including their
gitignored private records (checked against pinned checksums). Per-call
records stay in a gitignored `results/private/` folder; public results are
aggregates.
