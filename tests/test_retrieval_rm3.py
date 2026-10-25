"""RM3 expansion and the weighted BM25 query it uses. Skipped without the
optional `retrieval` extra."""

import ast
import math
from pathlib import Path

import pytest

pytest.importorskip("bm25s")

from oarp.envs.retrieval import rm3 as rm3_mod  # noqa: E402
from oarp.envs.retrieval.bm25 import BM25Index  # noqa: E402
from oarp.envs.retrieval.data import DEFAULT_DATA_DIR, load_scifact  # noqa: E402
from oarp.envs.retrieval.rm3 import RM3Config, expansion_weights, rm3_rank  # noqa: E402

CORPUS = {
    "d1": "aspirin reduces heart attack risk in adults",
    "d2": "aspirin and heart attack prevention trials",
    "d3": "myocardial infarction after cardiac surgery",
    "d4": "heart attack is also called myocardial infarction",
    "d5": "sleep duration and memory consolidation",
}
INDEX = BM25Index(CORPUS)


def test_weighted_query_with_token_counts_equals_plain_bm25() -> None:
    weights = {"aspirin": 2.0, "heart": 1.0}
    assert list(INDEX.weighted_scores(weights)) == pytest.approx(list(INDEX.scores("aspirin aspirin heart")))
    assert INDEX.rank_weighted({"unknownword": 1.0})[0][1] == 0.0


def test_lambda_one_reproduces_bm25() -> None:
    for query in ("aspirin heart attack", "myocardial infarction", "sleep and memory"):
        plain = [d for d, _ in INDEX.rank(query)]
        assert [d for d, _ in rm3_rank(INDEX, CORPUS, query, RM3Config(original_query_weight=1.0))] == plain


def test_expansion_weights_sum_to_one_and_mix_query_and_feedback_terms() -> None:
    config = RM3Config(fb_docs=2, fb_terms=3, original_query_weight=0.5)
    weights = expansion_weights(INDEX, CORPUS, "aspirin attack", config)
    assert math.isclose(sum(weights.values()), 1.0)
    assert weights["aspirin"] >= 0.25 and weights["attack"] >= 0.25  # half the mass is the original query
    assert len(set(weights) - {"aspirin", "attack"}) <= 3
    assert expansion_weights(INDEX, CORPUS, "the of and") == {}  # no vocabulary term


def test_expansion_can_bring_in_a_document_without_query_terms() -> None:
    # "heart" is a feedback term (d1, d2); d4 shares it but not "aspirin".
    plain = dict(INDEX.rank("aspirin"))
    expanded = dict(rm3_rank(INDEX, CORPUS, "aspirin", RM3Config(fb_docs=2, fb_terms=5)))
    assert plain["d4"] == 0.0 and expanded["d4"] > 0.0


def test_rm3_is_deterministic_and_config_is_validated() -> None:
    assert rm3_rank(INDEX, CORPUS, "heart attack") == rm3_rank(INDEX, CORPUS, "heart attack")
    with pytest.raises(ValueError):
        RM3Config(original_query_weight=1.5)


def test_rm3_module_never_touches_labels() -> None:
    tree = ast.parse(Path(rm3_mod.__file__).read_text())
    imported = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    retrieval = {name.split(".")[3] for name in imported if name.startswith("oarp.envs.retrieval.")}
    assert retrieval <= {"bm25"}


@pytest.mark.skipif(not (DEFAULT_DATA_DIR / "scifact" / "corpus.jsonl").is_file(), reason="SciFact not downloaded")
def test_lambda_one_reproduces_bm25_on_scifact_top_100() -> None:
    split = load_scifact(DEFAULT_DATA_DIR, "train")
    index = BM25Index(split.corpus)
    for query in list(split.queries.values())[:25]:
        plain = [d for d, _ in index.rank(query, 100)]
        same = [d for d, _ in rm3_rank(index, split.corpus, query, RM3Config(original_query_weight=1.0), 100)]
        assert same == plain
