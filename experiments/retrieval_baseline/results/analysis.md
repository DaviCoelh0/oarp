# R1: BM25 baseline on SciFact (train queries)

Model-free and deterministic. Corpus: 5183 documents. Queries in file: 1109; judged in the train split: 809; evaluated: 809; skipped: none. The test split was not loaded.

| nDCG@10 | Recall@10 | Recall@100 | MRR@10 |
|---|---|---|---|
| 0.6668 | 0.7869 | 0.9062 | 0.6347 |

## Where the first relevant document lands

| Bucket | Queries | Share |
|---|---|---|
| rank 1 | 452 | 55.9% |
| ranks 2-10 | 197 | 24.4% |
| ranks 11-100 | 89 | 11.0% |
| beyond 100 | 71 | 8.8% |

Queries with no relevant document in the top 10: 19.8% (triage guideline: judgment call; a guideline, not a decision).
Queries with several relevant documents: 77; of all queries, 26 found some but not all relevant documents in the top 10.
Queries with no word in the index vocabulary: 0.

BM25: {'k1': 1.5, 'b': 0.75, 'method': 'lucene', 'lower': True, 'stopwords': 'english', 'splitter': '(?u)\\b\\w\\w+\\b', 'stemmer': None}; bm25s 0.3.13, numpy 2.5.3. Timing: index 0.371 s, all queries 2.036 s.
