# R2a: reranking headroom on SciFact (train queries)

Model-free, deterministic, $0. Queries: 809 (tune 405, check 404). Test split not loaded. All numbers below use a PERFECT reranker or gate (they read labels): upper bounds, not results of any model.

## 1. Perfect reranker of BM25's top k

| k | first relevant in top k | nDCG@10 | Recall@10 | MRR@10 |
|---|---|---|---|---|
| 10 | 0.802 | 0.791 | 0.787 | 0.802 |
| 20 | 0.854 | 0.845 | 0.842 | 0.854 |
| 30 | 0.873 | 0.863 | 0.861 | 0.873 |
| 50 | 0.883 | 0.876 | 0.874 | 0.883 |
| 100 | 0.912 | 0.908 | 0.906 | 0.912 |

## 2. Oracle gate (rerank only when rank 1 is not relevant), perfect reranker, k = 20

| queries | rerank rate | S0 nDCG@10 | S1 (perfect) nDCG@10 | oracle gate nDCG@10 |
|---|---|---|---|---|
| all | 0.441 | 0.667 | 0.845 | 0.841 |
| tune | 0.435 | 0.668 | 0.840 | 0.836 |
| check | 0.448 | 0.666 | 0.851 | 0.847 |

## 3. Rule gate (relative gap < tau), perfect reranker; tau matched on tune, reported on check

| target rate | tau | check rerank rate | check nDCG@10 |
|---|---|---|---|
| 0.200 | 0.0512 | 0.193 | 0.729 |
| 0.300 | 0.0776 | 0.267 | 0.747 |
| 0.400 | 0.1114 | 0.342 | 0.767 |
| 0.500 | 0.1770 | 0.495 | 0.804 |
| 0.600 | 0.2568 | 0.629 | 0.829 |
| 0.800 | 0.4571 | 0.849 | 0.849 |
| 0.435 (oracle rate) | 0.1341 | 0.408 | 0.784 |

## 4. Does a BM25 signal predict "rank 1 not relevant"? (AUC, 0.5 = chance)

| signal | all | tune | check |
|---|---|---|---|
| relative_gap | 0.833 | 0.846 | 0.819 |
| raw_gap | 0.847 | 0.852 | 0.840 |
| top_score | 0.755 | 0.751 | 0.761 |

## 5. Estimated cost per query (Haiku 5.5; estimates, not measurements)

| call | mean prompt chars | est. input tokens | low effort $ | medium effort $ |
|---|---|---|---|---|
| rerank | 30249 | 7562 | 0.00079-0.00104 | 0.00089-0.00179 |
| gate | 4877 | 1219 | 0.00012-0.00022 | 0.00017-0.00062 |

Gate cost / rerank cost: low 0.12-0.29, medium 0.10-0.70. a gate saves money iff gate cost / rerank cost < 1 - its rerank rate; the oracle gate's rate on all dev queries is 0.441.
Input tokens at ~4 characters per token (heuristic); output tokens are assumptions: {'rerank': {'low': (60, 560), 'medium': (260, 2060)}, 'gate': {'low': (5, 205), 'medium': (105, 1005)}}. Prices as of 2026-10-19.
