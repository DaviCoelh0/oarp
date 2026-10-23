"""R2a strategy pieces with a scripted model: rerank and gate parsers and
pipelines, the offline combiner, the rule gate, label-reading analysis helpers,
AUC and the dev partition. No optional dependency, no network, no paid call."""

import ast
import math
from pathlib import Path

import pytest

from oarp.core.model import ScriptedModel
from oarp.core.types import ModelResponse, Usage
from oarp.envs.retrieval import analysis, gate, rerank, strategies
from oarp.envs.retrieval.data import dev_partition
from oarp.envs.retrieval.metrics import auc

CORPUS = {f"d{i}": f"abstract number {i}" for i in range(1, 7)}
BM25_ORDER = ["d3", "d1", "d5", "d2", "d6", "d4"]


def reply(text: str, usage: Usage | None = Usage(1000, 50)) -> ModelResponse:
    return ModelResponse(text=text, usage=usage, model="claude-haiku-5-5")


# --- rerank parser -------------------------------------------------------------

def test_valid_permutation() -> None:
    p = rerank.parse_ranking("thinking...\nRANKING: 3, 1, 2", 3)
    assert (p.order, p.status, p.duplicates, p.out_of_range, p.missing) == ([2, 0, 1], "valid", 0, 0, 0)


def test_repairs_are_counted() -> None:
    p = rerank.parse_ranking("RANKING: 2, 2, 9, 0, 4", 4)
    # 2 kept once; 9 and 0 out of range; 4 kept; 1 and 3 missing, appended in BM25 order.
    assert p.order == [1, 3, 0, 2]
    assert (p.status, p.duplicates, p.out_of_range, p.missing) == ("repaired", 1, 2, 2)


@pytest.mark.parametrize("text", ["", "I think 2 is best", "RANKING:", "RANKING: none", "RANKING: 7, 8"])
def test_failed_answers_keep_bm25_order(text: str) -> None:
    p = rerank.parse_ranking(text, 3)
    assert p.order == [0, 1, 2] and p.status == "failed"


def test_last_ranking_line_counts_and_case_is_ignored() -> None:
    assert rerank.parse_ranking("RANKING: 1, 2\nranking: 2, 1", 2).order == [1, 0]


# --- rerank pipeline -----------------------------------------------------------

def test_rerank_reorders_top_k_and_keeps_the_rest() -> None:
    model = ScriptedModel([reply("RANKING: 3, 2, 1")])
    record = rerank.rerank(model, "q1", "claim text", BM25_ORDER, CORPUS, rerank.RerankConfig(k=3))
    assert record.ranking == ["d5", "d1", "d3", "d2", "d6", "d4"]
    assert record.status == "valid" and record.usage == Usage(1000, 50) and record.model == "claude-haiku-5-5"


def test_rerank_prompt_has_query_and_texts_but_no_ids() -> None:
    model = ScriptedModel([reply("RANKING: 1, 2, 3")])
    rerank.rerank(model, "q1", "claim text", BM25_ORDER, CORPUS, rerank.RerankConfig(k=3))
    messages, tools = model.calls[0]
    user = messages[1].content
    assert tools == [] and "claim text" in user
    assert "[1] abstract number 3" in user and "[3] abstract number 5" in user and "[4]" not in user
    assert not any(doc_id in m.content for m in messages for doc_id in CORPUS)


def test_identity_answer_reproduces_bm25_order() -> None:
    model = ScriptedModel([reply("RANKING: 1, 2, 3, 4, 5, 6")])
    record = rerank.rerank(model, "q1", "q", BM25_ORDER, CORPUS, rerank.RerankConfig(k=20))
    assert record.ranking == BM25_ORDER and record.copied_presentation_order and record.matches_bm25_order


def test_documents_are_truncated() -> None:
    messages = rerank.rerank_messages("q", ["x" * 50], rerank.RerankConfig(k=1, max_doc_chars=10))
    assert "[1] " + "x" * 10 + " …" in messages[1].content


def test_model_error_keeps_bm25_order() -> None:
    record = rerank.rerank(ScriptedModel([]), "q1", "q", BM25_ORDER, CORPUS)
    assert record.status == "model_error" and record.ranking == BM25_ORDER and record.usage is None
    assert "ScriptExhausted" in record.error


# --- shuffled presentation --------------------------------------------------

SHUFFLED = rerank.RerankConfig(k=6, shuffle=True)


def answer_in_bm25_order(query_id: str, k: int) -> str:
    shown = rerank.presentation_order(query_id, k)
    return "RANKING: " + ", ".join(str(shown.index(i) + 1) for i in range(k))


def test_presentation_order_is_a_deterministic_per_query_permutation() -> None:
    first = rerank.presentation_order("q1", 20)
    assert sorted(first) == list(range(20)) and first != list(range(20))
    assert rerank.presentation_order("q1", 20) == first
    assert rerank.presentation_order("q2", 20) != first
    # Independent of the order in which queries are processed (no shared random state).
    ids = ["q3", "q1", "q2"]
    assert {q: rerank.presentation_order(q, 20) for q in ids} == {q: rerank.presentation_order(q, 20) for q in reversed(ids)}


def test_shuffle_inverts_back_to_document_ids() -> None:
    model = ScriptedModel([reply(answer_in_bm25_order("q1", 6))])
    record = rerank.rerank(model, "q1", "claim", BM25_ORDER, CORPUS, SHUFFLED)
    assert record.ranking == BM25_ORDER and record.status == "valid"
    assert record.matches_bm25_order and not record.copied_presentation_order


def test_shuffle_prompt_follows_the_permutation_and_hides_ids() -> None:
    shown = rerank.presentation_order("q1", 6)
    model = ScriptedModel([reply("RANKING: 1, 2, 3, 4, 5, 6")])
    record = rerank.rerank(model, "q1", "claim", BM25_ORDER, CORPUS, SHUFFLED)
    user = model.calls[0][0][1].content
    for j, position in enumerate(shown, 1):
        assert f"[{j}] {CORPUS[BM25_ORDER[position]]}" in user
    assert not any(doc_id in m.content for m in model.calls[0][0] for doc_id in CORPUS)
    # Copying the presentation order gives the shuffled order, not BM25's.
    assert record.ranking == [BM25_ORDER[i] for i in shown]
    assert record.copied_presentation_order and not record.matches_bm25_order


def test_repairs_under_shuffle_append_missing_in_bm25_order() -> None:
    shown = rerank.presentation_order("q1", 6)
    model = ScriptedModel([reply("RANKING: 2, 2, 99")])
    record = rerank.rerank(model, "q1", "claim", BM25_ORDER, CORPUS, SHUFFLED)
    first = BM25_ORDER[shown[1]]
    assert record.ranking == [first] + [d for d in BM25_ORDER if d != first]
    assert (record.status, record.duplicates, record.out_of_range, record.missing) == ("repaired", 1, 1, 5)


@pytest.mark.parametrize("script", [[reply("no ranking here")], []])
def test_failures_under_shuffle_keep_bm25_order(script) -> None:
    record = rerank.rerank(ScriptedModel(script), "q1", "claim", BM25_ORDER, CORPUS, SHUFFLED)
    assert record.ranking == BM25_ORDER and record.status in {"failed", "model_error"}
    assert record.matches_bm25_order and not record.copied_presentation_order


def test_shuffle_is_recorded_in_provenance() -> None:
    off, on = rerank.provenance(rerank.RerankConfig()), rerank.provenance(SHUFFLED)
    assert off["rerank_config"]["shuffle"] is False and off["rerank_shuffle_seed"] is None
    assert on["rerank_config"]["shuffle"] is True and on["rerank_shuffle_seed"] == "oarp-r2-rerank-shuffle-v1:"


# --- gate ----------------------------------------------------------------------

@pytest.mark.parametrize("text, status", [
    ("ANSWER: yes", "yes"), ("answer: NO", "no"), ("ANSWER: no\nANSWER: yes", "yes"),
    ("yes", "unparseable"), ("", "unparseable"), ("ANSWER: maybe", "unparseable"), ("ANSWER: yesno", "unparseable"),
])
def test_gate_parser(text: str, status: str) -> None:
    assert gate.parse_gate(text) == status


def test_gate_shows_exactly_the_top_three() -> None:
    model = ScriptedModel([reply("ANSWER: yes", Usage(300, 5))])
    record = gate.gate(model, "claim text", BM25_ORDER, CORPUS)
    assert record.keep_bm25 and record.status == "yes" and record.usage == Usage(300, 5)
    user = model.calls[0][0][1].content
    assert "claim text" in user and "[3] abstract number 5" in user and "[4]" not in user
    assert "abstract number 2" not in user and not any(d in user for d in CORPUS)
    with pytest.raises(ValueError):
        gate.gate_messages("q", ["a", "b"])


@pytest.mark.parametrize("script", [[reply("ANSWER: no")], [reply("not sure")], []])
def test_gate_reranks_on_no_unparseable_or_error(script) -> None:
    record = gate.gate(ScriptedModel(script), "q", BM25_ORDER, CORPUS)
    assert not record.keep_bm25
    assert record.status in {"no", "unparseable", "model_error"}


def test_flip_rate() -> None:
    assert gate.flip_rate({"a": [True, True], "b": [True, False], "c": [False, False, False]}) == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        gate.flip_rate({"a": [True]})


# --- strategies (label-free) ---------------------------------------------------

def test_bm25_signals() -> None:
    assert strategies.bm25_signals([("a", 4.0), ("b", 3.0)]) == {"relative_gap": 0.25, "raw_gap": 1.0, "top_score": 4.0}
    assert strategies.bm25_signals([("a", 0.0), ("b", 0.0)])["relative_gap"] == 0.0  # s1 = 0: rerank
    assert strategies.bm25_signals([("a", 2.0)]) == {"relative_gap": 1.0, "raw_gap": 2.0, "top_score": 2.0}


def test_rule_gate_extremes_and_threshold() -> None:
    gaps = {"a": 0.0, "b": 0.1, "c": 0.5}
    assert not any(strategies.rule_gate(gaps, -math.inf).values())  # S0
    assert all(strategies.rule_gate(gaps, math.inf).values())  # S1
    assert strategies.rule_gate(gaps, 0.1) == {"a": True, "b": False, "c": False}  # strictly below tau


def test_tau_for_rate() -> None:
    gaps = [0.0, 0.1, 0.2, 0.3]
    assert strategies.tau_for_rate(gaps, 0.5) == 0.2  # 0.0 and 0.1 below
    assert strategies.tau_for_rate(gaps, 0.0) == -math.inf
    assert strategies.tau_for_rate(gaps, 1.0) == math.inf
    assert strategies.tau_for_rate([0.1, 0.1, 0.1, 0.1], 0.5) == -math.inf  # ties: rate 0 and 1 equally far; fewer reranks


def test_combiner_extremes() -> None:
    bm25 = {"q1": ["a", "b"], "q2": ["c", "d"]}
    reranked = {"q1": ["b", "a"], "q2": ["d", "c"]}
    always_yes = strategies.gate_decisions({"q1": True, "q2": True})
    always_no = strategies.gate_decisions({"q1": False, "q2": False})
    assert strategies.combine(bm25, reranked, always_yes) == bm25  # S0
    assert strategies.combine(bm25, reranked, always_no) == reranked  # S1
    assert strategies.combine(bm25, reranked, {"q1": True, "q2": False}) == {"q1": ["b", "a"], "q2": ["c", "d"]}
    with pytest.raises(ValueError):
        strategies.combine(bm25, {"q1": ["a"]}, always_yes)


def test_usage_totals() -> None:
    model = ScriptedModel([reply("RANKING: 1", Usage(1000, 100)), reply("RANKING: 1", Usage(2000, 300))])
    records = [rerank.rerank(model, "q1", "q", BM25_ORDER, CORPUS, rerank.RerankConfig(k=1)) for _ in range(2)]
    totals = strategies.usage_totals(records, "claude-haiku-5-5")
    assert (totals["calls"], totals["input_tokens"], totals["output_tokens"]) == (2, 3000, 400)
    assert totals["estimated_usd"] == pytest.approx(3000 * 0.10 / 1e6 + 400 * 0.50 / 1e6)
    records.append(rerank.rerank(ScriptedModel([]), "q1", "q", BM25_ORDER, CORPUS))
    totals = strategies.usage_totals(records, "claude-haiku-5-5")
    assert totals["calls_without_usage"] == 1 and totals["estimated_usd"] is None  # never guessed


# --- analysis helpers (read labels) --------------------------------------------

def test_oracle_rerank_moves_relevant_within_top_k_only() -> None:
    judged = {"d5": 1, "d4": 2, "d2": 0}
    assert analysis.oracle_rerank(BM25_ORDER, judged, 3) == ["d5", "d3", "d1", "d2", "d6", "d4"]
    assert analysis.oracle_rerank(BM25_ORDER, judged, 6) == ["d4", "d5", "d3", "d1", "d2", "d6"]


def test_oracle_gate_and_precision_recall() -> None:
    bm25 = {"a": ["x", "y"], "b": ["y", "x"], "c": ["x", "y"], "d": ["y", "x"]}
    qrels = {q: {"x": 1} for q in bm25}  # rank 1 relevant for a and c
    assert analysis.oracle_gate(bm25, qrels) == {"a": False, "b": True, "c": False, "d": True}
    keep = {"a": True, "b": True, "c": False, "d": False}  # one hit, one false yes, one miss
    pr = analysis.gate_precision_recall(keep, qrels, bm25)
    assert (pr["precision"], pr["recall"], pr["said_yes"], pr["rank1_relevant"]) == (0.5, 0.5, 2, 2)
    assert analysis.gate_precision_recall({q: False for q in bm25}, qrels, bm25)["precision"] is None


def test_mixed_mean_and_choose_tau() -> None:
    without, with_ = {"a": 1.0, "b": 0.0, "c": 0.5}, {"a": 0.0, "b": 1.0, "c": 0.5}
    gaps = {"a": 0.9, "b": 0.1, "c": 0.4}
    assert analysis.mixed_mean({"a": False, "b": True, "c": True}, without, with_) == pytest.approx(2.5 / 3)
    # Rerank b only (tau in (0.1, 0.4]) is best; c is a tie, so the smaller tau (0.4) wins over 0.9.
    assert analysis.choose_tau(gaps, without, with_) == 0.4


# --- AUC and partition ---------------------------------------------------------

def test_auc_by_hand() -> None:
    # Positives 0.35, 0.8; negatives 0.1, 0.4: pairs won 0.35>0.1, 0.8>0.1, 0.8>0.4 -> 3 of 4.
    assert auc([0.1, 0.4, 0.35, 0.8], [False, False, True, True]) == 0.75
    assert auc([1.0, 1.0], [True, False]) == 0.5  # a tie counts one half
    assert auc([0.0, 1.0], [True, False]) == 0.0
    with pytest.raises(ValueError):
        auc([1.0, 2.0], [True, True])


def test_dev_partition_is_deterministic_disjoint_and_balanced() -> None:
    ids = [f"q{i}" for i in range(11)]
    split = dev_partition(ids)
    assert split == dev_partition(list(reversed(ids)))  # independent of input order
    assert len(split["tune"]) == 6 and len(split["check"]) == 5
    assert set(split["tune"]).isdisjoint(split["check"]) and set(split["tune"]) | set(split["check"]) == set(ids)
    assert dev_partition(ids, seed="other") != split


# --- the strategy path never reads labels --------------------------------------

@pytest.mark.parametrize("module", [rerank, gate, strategies])
def test_strategy_modules_never_touch_labels(module) -> None:
    tree = ast.parse(Path(module.__file__).read_text())
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imported |= {f"{n.module}.{a.name}" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    retrieval = {name.split(".")[3] for name in imported if name.startswith("oarp.envs.retrieval.")}
    assert retrieval <= {"rerank", "gate", "strategies"}  # never analysis, data (qrels) or metrics
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        a.arg for n in ast.walk(tree) if isinstance(n, ast.arguments) for a in n.args}
    assert not names & {"qrels", "judged", "relevant"}
