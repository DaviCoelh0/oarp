# S3a: reachability and RM3 on SciFact (tune half)

Model-free, deterministic, $0. Tune queries: 405. Check half and test split not used. Sections 1-2 read labels (oracle quantities); section 3 is a label-free method scored with labels.

## 1. Strata by BM25's first relevant rank (first search = top 20)

Within top 20: 342; beyond: 63 (ranks 21-100: 26, 101-1000: 17, beyond 1000 or none: 20).

## 2. Title reachability for the 63 beyond-20 queries (oracle, optimistic)

| query = a relevant document's own title | within top 10 | within top 20 |
|---|---|---|
| that document is found | 1.000 | 1.000 |
| any relevant document is found | 1.000 | 1.000 |

Best over the query's relevant documents. The title is part of the indexed text (close to a known-item search), so this overstates what a model's reformulation can reach; a failure does not prove that no query works.

## 3. RM3 (fixed defaults, not tuned) vs BM25, all tune queries

| metric | BM25 | RM3 | RM3 - BM25 [95% CI] |
|---|---|---|---|
| ndcg@10 | 0.668 | 0.624 | -0.044 [-0.065, -0.024] |
| recall@20 | 0.839 | 0.839 | +0.000 [-0.022, +0.021] |
| recall@100 | 0.906 | 0.927 | +0.021 [+0.005, +0.038] |

By stratum: of the beyond-20 queries, RM3 puts a relevant document into its top 10 for 0.032 and into its top 20 for 0.159; of the within-20 queries, RM3 loses every relevant document from its top 20 for 0.029. Mean nDCG@10 change: within -0.054, beyond 0.011.
RM3 settings: {'fb_docs': 10, 'fb_terms': 10, 'original_query_weight': 0.5}.
