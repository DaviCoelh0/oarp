"""Retrieval metrics against hand-calculated values (no optional dependency needed)."""

import math

import pytest

from oarp.envs.retrieval.metrics import (
    first_relevant_rank,
    headroom_bucket,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)

L3, L5 = math.log2(3), math.log2(5)


def test_perfect_single_relevant() -> None:
    assert ndcg_at_k(["a", "x"], {"a": 1}, 10) == 1.0


def test_single_relevant_at_rank_two() -> None:
    assert ndcg_at_k(["x", "a"], {"a": 1}, 10) == pytest.approx(1 / L3)  # 0.6309


def test_two_relevant_documents() -> None:
    ranking, judged = ["x", "a", "y", "b"], {"a": 1, "b": 1}
    # DCG = 1/log2(3) + 1/log2(5); ideal DCG = 1 + 1/log2(3)
    assert ndcg_at_k(ranking, judged, 10) == pytest.approx((1 / L3 + 1 / L5) / (1 + 1 / L3))
    # Cut at 3: b falls out of the DCG, but the ideal still has both relevant documents.
    assert ndcg_at_k(ranking, judged, 3) == pytest.approx((1 / L3) / (1 + 1 / L3))
    # Cut at 1: nothing relevant in the top 1; the ideal DCG@1 is 1.
    assert ndcg_at_k(ranking, judged, 1) == 0.0
    assert recall_at_k(ranking, judged, 2) == 0.5  # denominator = all relevant documents
    assert recall_at_k(ranking, judged, 4) == 1.0
    assert recall_at_k(ranking, judged, 1) == 0.0


def test_graded_relevance() -> None:
    # Ranking b (grade 1) then a (grade 2): DCG = 1 + 2/log2(3); ideal = 2 + 1/log2(3).
    assert ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, 10) == pytest.approx((1 + 2 / L3) / (2 + 1 / L3))


def test_ideal_dcg_is_truncated_at_k() -> None:
    judged = {"a": 1, "b": 1, "c": 1}
    assert ndcg_at_k(["a", "b"], judged, 2) == 1.0  # ideal DCG@2 uses only 2 of the 3 relevant


def test_zero_grade_and_unjudged_count_as_not_relevant() -> None:
    judged = {"a": 1, "z": 0}
    assert ndcg_at_k(["z", "u", "a"], judged, 10) == pytest.approx(1 / math.log2(4))
    assert first_relevant_rank(["z", "u", "a"], judged) == 3
    assert recall_at_k(["z", "u"], judged, 10) == 0.0


def test_mrr_and_first_rank() -> None:
    assert mrr_at_k(["x", "y", "a"], {"a": 1}, 10) == pytest.approx(1 / 3)
    assert mrr_at_k(["x", "y", "a"], {"a": 1}, 2) == 0.0
    assert first_relevant_rank(["x", "y"], {"a": 1}) is None


def test_query_without_relevant_documents_is_an_error() -> None:
    with pytest.raises(ValueError, match="no relevant document"):
        ndcg_at_k(["a"], {"a": 0}, 10)


@pytest.mark.parametrize(("rank", "bucket"), [
    (1, "rank 1"), (2, "ranks 2-10"), (10, "ranks 2-10"), (11, "ranks 11-100"),
    (100, "ranks 11-100"), (101, "beyond 100"), (None, "beyond 100"),
])
def test_headroom_buckets(rank, bucket: str) -> None:
    assert headroom_bucket(rank) == bucket
