"""Retrieval metrics for one query, given its ranked doc ids and its judgments.

- A document is relevant if its grade is > 0. Unjudged documents count as
  non-relevant (BEIR convention), and so do documents judged with grade 0.
- nDCG@k: gain = grade, discount = 1 / log2(rank + 1); the ideal DCG uses all
  relevant documents of the query, best grades first, truncated at k. (For
  binary grades, linear and exponential gains give the same value.)
- Recall@k: relevant documents in the top k / all relevant documents.
- MRR@k: 1 / rank of the first relevant document if it is within k, else 0.
- first_relevant_rank: 1-based rank of the first relevant document in the
  full ranking, or None if none was retrieved.
"""

import math

HEADROOM_BUCKETS = ("rank 1", "ranks 2-10", "ranks 11-100", "beyond 100")


def relevant(judged: dict[str, int]) -> dict[str, int]:
    rel = {doc: grade for doc, grade in judged.items() if grade > 0}
    if not rel:
        raise ValueError("query has no relevant document; it should have been skipped")
    return rel


def ndcg_at_k(ranking: list[str], judged: dict[str, int], k: int) -> float:
    rel = relevant(judged)
    dcg = sum(rel.get(doc, 0) / math.log2(rank + 1) for rank, doc in enumerate(ranking[:k], 1))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(grade / math.log2(rank + 1) for rank, grade in enumerate(ideal, 1))
    return dcg / idcg


def recall_at_k(ranking: list[str], judged: dict[str, int], k: int) -> float:
    rel = relevant(judged)
    return sum(doc in rel for doc in ranking[:k]) / len(rel)


def first_relevant_rank(ranking: list[str], judged: dict[str, int]) -> int | None:
    rel = relevant(judged)
    return next((rank for rank, doc in enumerate(ranking, 1) if doc in rel), None)


def mrr_at_k(ranking: list[str], judged: dict[str, int], k: int) -> float:
    rank = first_relevant_rank(ranking[:k], judged)
    return 0.0 if rank is None else 1.0 / rank


def headroom_bucket(rank: int | None) -> str:
    if rank == 1:
        return HEADROOM_BUCKETS[0]
    if rank is not None and rank <= 10:
        return HEADROOM_BUCKETS[1]
    if rank is not None and rank <= 100:
        return HEADROOM_BUCKETS[2]
    return HEADROOM_BUCKETS[3]
