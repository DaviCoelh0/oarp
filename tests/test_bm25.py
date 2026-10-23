"""BM25 wrapper: deterministic full ranking, ties broken by document id even
across the top-k cut-off. Skipped without the optional `retrieval` extra."""

import pytest

pytest.importorskip("bm25s")

from oarp.envs.retrieval.bm25 import BM25Config, BM25Index, provenance  # noqa: E402

CORPUS = {
    "d1": "aspirin reduces heart attack risk",
    "d2": "statins lower cholesterol",
    "d3": "vitamin deficiency and fractures",
    "d4": "coffee and heart rhythm",
}


def test_rare_shared_word_ranks_first() -> None:
    assert BM25Index(CORPUS).rank("cholesterol", 1)[0][0] == "d2"


def test_full_ranking_covers_the_corpus_once() -> None:
    ranking = BM25Index(CORPUS).rank("heart")
    assert sorted(d for d, _ in ranking) == sorted(CORPUS)
    assert {d for d, _ in ranking[:2]} == {"d1", "d4"}  # the two documents containing "heart"


def test_two_builds_give_identical_rankings() -> None:
    assert BM25Index(CORPUS).rank("heart attack risk") == BM25Index(CORPUS).rank("heart attack risk")


def test_ties_across_the_cutoff_are_broken_by_doc_id() -> None:
    # Five identical documents tie; inserted out of id order. The top 3 must be the 3 lowest ids.
    corpus = {"e5": "same text", "e2": "same text", "e9": "same text", "e1": "same text", "e7": "same text",
              "x": "other words"}
    top3 = BM25Index(corpus).rank("same text", 3)
    assert [d for d, _ in top3] == ["e1", "e2", "e5"]
    assert len({s for _, s in top3}) == 1  # really tied


def test_query_without_indexed_words_ranks_by_doc_id() -> None:
    ranking = BM25Index(CORPUS).rank("zzz unknownword", 4)
    assert [d for d, _ in ranking] == ["d1", "d2", "d3", "d4"] and all(s == 0.0 for _, s in ranking)
    assert [d for d, _ in BM25Index(CORPUS).rank("", 2)] == ["d1", "d2"]


def test_tokenizer_settings() -> None:
    index = BM25Index(CORPUS)
    # Lowercased, stopwords dropped, and no stemming: "fracture" does not match "fractures".
    assert index.query_tokens("The HEART and fracture!") == ["heart"]
    assert BM25Config().stemmer is None and BM25Config().k1 == 1.5 and BM25Config().b == 0.75


def test_provenance_records_versions_and_config() -> None:
    info = provenance(BM25Config())
    assert info["bm25s"] == "0.3.13" and info["numpy"] == "2.5.3"
    assert info["bm25_config"]["method"] == "lucene"


def test_empty_corpus_is_refused() -> None:
    with pytest.raises(ValueError, match="empty corpus"):
        BM25Index({})
