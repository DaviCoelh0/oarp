# S3a: reachability and an RM3 baseline (free, no model calls)

Before any paid run of the search agent (S3), this asks how much a second
BM25 search could add at all, and how a classic model-free "search again"
does: RM3 pseudo-relevance feedback.

    venv/bin/python experiments/retrieval_s3_reachability/run_analysis.py

Deterministic, $0, about 6 s. SciFact train queries, **tune half only (405)**;
the check half and the test split are not used. Outputs:
[`results/analysis.md`](results/analysis.md), `results/summary.json`
(aggregates, with git commit and settings); per-query details in the
gitignored `results/private/`.

## Setup

- BM25 as in the R1 baseline (`bm25s` 0.3.13, k1 1.5, b 0.75, Lucene, English
  stopwords, no stemming).
- Strata: BM25's first relevant rank within or beyond 20, the agent's forced
  first search.
- Title reachability (an oracle that reads labels): for each beyond-20 query,
  BM25 with each relevant document's own title as the query.
- RM3 (`src/oarp/envs/retrieval/rm3.py`), fixed in advance at the common
  defaults (10 feedback documents, 10 expansion terms, original-query weight
  0.5), **not tuned**. RM3-style, written in oarp; not a reimplementation of
  any toolkit's RM3.

## Results (tune, n = 405)

| | |
|---|---|
| Strata | 342 within top 20, 63 beyond (26 at ranks 21-100, 17 at 101-1000, 20 beyond 1000) |
| Title reachability, beyond-20 queries | 63/63 find the titled document in the top 10 |
| RM3 - BM25, nDCG@10 | -0.044 [-0.065, -0.024] (BM25 0.668, RM3 0.624) |
| RM3 - BM25, Recall@20 | +0.000 [-0.022, +0.021] |
| RM3 - BM25, Recall@100 | +0.021 [+0.005, +0.038] |
| RM3 rescues beyond-20 queries | into its top 10: 2/63 (3.2%); into its top 20: 10/63 (15.9%) |
| RM3 harm, within-20 queries | every relevant document leaves its top 20: 10/342 (2.9%) |

95% intervals: paired percentile bootstrap over queries (10,000 resamples,
seed 0).

## Interpretation and limitations

- **Title reachability is uninformative.** The title is part of each indexed
  document, so a title query is a known-item search, and it reaches every
  document. It shows that no beyond-20 document is unreachable by BM25 in
  principle, but it gives no realistic ceiling for what a reformulation
  written from the claim alone can reach. (The proposal for this analysis
  expected an informative bound; it is not one.)
- **RM3 is not a good fixed reformulation here.** It hurts nDCG@10, leaves
  Recall@20 unchanged and helps only deep recall (Recall@100). It brings a
  relevant document into the top 20 for about one in six beyond-20 queries,
  and loses one for about 3% of within-20 queries. As a free comparison for
  the agent: a model search would need to rescue more than this, at a lower
  harm rate, to be worth its cost.
- One dataset, the tune half only, untuned defaults; a different RM3 setting
  could do better (tuning it would be a separate decision).
