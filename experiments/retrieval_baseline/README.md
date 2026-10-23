# R1: BM25 baseline on SciFact (development queries)

The first measurement in oarp's retrieval environment, on 2026-10-23. **Plain
BM25, no model calls, $0.** Its purpose is to check that retrieval quality
can be measured reliably, and to find out where an adaptive retrieval policy
could have room to improve, before any agent is built.

## Setup

| | |
|---|---|
| Data | BEIR SciFact (`scifact.zip`, 2,816,079 bytes, MD5 `5f7d1de6…efca1` as published by BEIR, SHA-256 `536e1444…0165` pinned in `src/oarp/envs/retrieval/data.py`). Not included in this repository: download it with `python -m oarp.envs.retrieval.data`. Licenses of the original release: claims and annotations CC BY 4.0, abstracts ODC-By 1.0. |
| Split | the **809 train queries** (all have at least one relevant document) serve as development data. The 300 test queries are held out and were **not loaded**. |
| Retriever | `bm25s` 0.3.13 (numpy 2.5.3), Lucene-style BM25, k1 = 1.5, b = 0.75; document text = title + abstract; lowercase, English stopwords, default word splitter, **no stemming**. Every query scores the whole corpus (5,183 documents); ties are broken by document id before any cut-off, so rankings are deterministic. |
| Metrics | nDCG@10, Recall@10, Recall@100, MRR@10, and the rank of the first relevant document. Unjudged documents count as non-relevant (BEIR convention). Labels are binary here, with 1.14 relevant documents per query on average. |

Reproduce: `pip install -e ".[retrieval]"`, download as above, then
`venv/bin/python experiments/retrieval_baseline/run_baseline.py`.

## Results

| nDCG@10 | Recall@10 | Recall@100 | MRR@10 |
|---|---|---|---|
| 0.667 | 0.787 | 0.906 | 0.635 |

Where the first relevant document lands (809 queries):

| Rank | Queries | Share |
|---|---|---|
| 1 | 452 | 55.9% |
| 2–10 | 197 | 24.4% |
| 11–100 | 89 | 11.0% |
| beyond 100 | 71 | 8.8% |

- 19.8% of queries have no relevant document in the top 10.
- 77 queries have more than one relevant document; 26 of them found some
  but not all of them in the top 10.
- Indexing took 0.37 s and all 809 queries 2.0 s on a laptop.

`results/summary.json` and `results/analysis.md` hold the aggregates. Per-query
details are derived from the benchmark labels and stay in a gitignored
`results/private/` folder.

## Interpretation and limitations

- **There is room to improve, mostly in ranking.** For 35.4% of queries the
  first relevant document is retrieved but not at rank 1 (ranks 2–100), which
  a better ranking of the existing candidates could address. For 8.8% it is
  not in the top 100 at all, which only a different query or retrieval method
  could recover.
- **The pre-stated triage guideline lands in its "judgment call" band:** 19.8%
  with no relevant document in the top 10, against guidelines of ≥ 20%
  (suggests an adaptive policy) and < 10% (suggests a harder dataset). The
  guideline was a triage aid, not a decision rule.
- **Not comparable to published BEIR numbers,** which use the test split and
  a different BM25 implementation (Elasticsearch, with stemming).
- **Sparse labels.** With about one relevant document per query, a single
  query's score is coarse; only aggregates over many queries are informative.
- **Untuned.** BM25 parameters and tokenization are library defaults; no
  setting was chosen using these results.
