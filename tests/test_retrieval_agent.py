"""S3a: the retrieval agent graph with a scripted model (no paid calls).

Covers the builder's structure, the forced first search, S0 and S1
reproduction, model searches and their budget, rejected calls, stop reasons
with their fallback rankings, the repair rule, accounting, and that no dataset
id reaches the model. Skipped without the `retrieval` and `langgraph` extras."""

import ast
import json
from dataclasses import asdict
from pathlib import Path

import pytest

pytest.importorskip("bm25s")
pytest.importorskip("langgraph")

from oarp.core.model import ScriptedModel  # noqa: E402
from oarp.core.pricing import estimate_cost  # noqa: E402
from oarp.core.types import ModelResponse, ToolCall, Usage  # noqa: E402
from oarp.envs.retrieval import agent as agent_mod  # noqa: E402
from oarp.envs.retrieval.agent import (  # noqa: E402
    AgentConfig,
    build_agent_graph,
    config_sha256,
    parse_handles,
    run_episode,
)
from oarp.envs.retrieval.bm25 import BM25Index  # noqa: E402
from oarp.envs.retrieval.rerank import RerankConfig, rerank, rerank_messages  # noqa: E402

MODEL = "claude-haiku-5-5"
CLAIM = "alpha protein binds receptor"
TOPICS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]


def make_corpus() -> dict[str, str]:
    """30 documents with opaque ids. Documents 0-23 mention the claim's words
    (with varying weight), 24-29 do not; doc 11 is longer than 2,000 characters."""
    corpus = {}
    for i in range(30):
        words = (["alpha"] * (1 + i % 4) + ["protein"] * (i % 3) + ["receptor"]) if i < 24 else []
        text = " ".join(words + [TOPICS[i % 6], f"marker{i}", "kinase" if i % 5 == 0 else "cell"])
        if i == 11:
            text += " the" * 600  # stopwords: longer text, same BM25 length
        corpus[f"sf{i:04d}x"] = f"Title {i}. {text}"
    return corpus


CORPUS = make_corpus()
INDEX = BM25Index(CORPUS)
BM25 = [d for d, _ in INDEX.rank(CLAIM)]


def call(name: str, i: int = 0, **arguments) -> ToolCall:
    return ToolCall(f"c{i}", name, arguments)


def respond(*calls: ToolCall, text: str = "", usage: Usage = Usage(1000, 100)) -> ModelResponse:
    return ModelResponse(text=text, tool_calls=calls, usage=usage, model=MODEL)


def finish(answer: str, i: int = 9) -> ModelResponse:
    return respond(call("finish", i, answer=answer))


def ranking_line(handles) -> str:
    return "RANKING: " + ", ".join(handles)


def episode(responses, config: AgentConfig = AgentConfig(), index: BM25Index = INDEX):
    model = ScriptedModel(responses)
    record = run_episode(build_agent_graph(config, model, index, CORPUS), "q1", CLAIM)
    return record, model


def tool_messages(model: ScriptedModel) -> list[str]:
    return [m.content for m in model.calls[-1][0] if m.role == "tool"]


# --- builder and config ------------------------------------------------------

def test_builder_adds_rerank_node_only_for_forced_rerank() -> None:
    model = ScriptedModel([])
    plain = build_agent_graph(AgentConfig(), model, INDEX, CORPUS).graph.get_graph().nodes
    forced = build_agent_graph(AgentConfig(final_order="forced_rerank"), model, INDEX, CORPUS).graph.get_graph().nodes
    assert set(forced) - set(plain) == {"rerank"}
    assert {"forced_search", "model", "tools", "assemble"} <= set(plain)


def test_config_hash_is_stable_and_config_is_validated() -> None:
    assert config_sha256(AgentConfig()) == config_sha256(AgentConfig())
    assert config_sha256(AgentConfig()) != config_sha256(AgentConfig(final_order="forced_rerank"))
    with pytest.raises(ValueError):
        AgentConfig(final_order="model")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        AgentConfig(max_searches=0)


# --- forced first search, S0 and S1 reproduction ---------------------------------

def test_forced_search_is_bm25_top_20_at_2000_characters() -> None:
    record, model = episode([finish(ranking_line(f"D{i}" for i in range(1, 21)))])
    assert record.candidates == BM25[:20]
    assert record.searches == [{"query": CLAIM, "source": "harness", "handles": [f"D{i}" for i in range(1, 21)],
                                "new_handles": [f"D{i}" for i in range(1, 21)]}]
    first = model.calls[0][0][1].content
    for i, doc_id in enumerate(BM25[:20], 1):
        text = CORPUS[doc_id]
        shown = text if len(text) <= 2000 else text[:2000].rstrip() + " …"
        assert f"[D{i}] {shown}\n" in first + "\n"
    assert "sf0011x" in BM25[:20] and len(CORPUS["sf0011x"]) > 2000
    assert [t.name for t in model.calls[0][1]] == ["search", "finish"]


def test_finishing_in_bm25_order_reproduces_s0() -> None:
    record, _ = episode([finish(ranking_line(f"D{i}" for i in range(1, 21)))])
    assert record.ranking == BM25 and record.agent_ranking_status == "valid"
    assert record.stop_reason == "finished" and record.model_calls == 1


def test_forced_rerank_without_extra_search_reproduces_s1() -> None:
    answer = "RANKING: " + ", ".join(str(n) for n in [3, 1, 2, *range(4, 21)])
    rerank_response = ModelResponse(text=answer, usage=Usage(2000, 50), model=MODEL)
    record, model = episode([finish("RANKING: D1"), rerank_response], AgentConfig(final_order="forced_rerank"))
    s1_request = rerank_messages(CLAIM, [CORPUS[d] for d in BM25[:20]], RerankConfig())
    assert model.calls[1][0] == s1_request and model.calls[1][1] == []
    s1 = rerank(ScriptedModel([rerank_response]), "q1", CLAIM, BM25, CORPUS, RerankConfig())
    assert record.ranking == s1.ranking
    assert record.agent_ranking[0] == BM25[0]  # the agent's own order is still recorded
    assert record.rerank["status"] == "valid"


def test_agent_prompt_does_not_depend_on_final_order() -> None:
    _, plain = episode([finish("RANKING: D1")])
    _, forced = episode([finish("RANKING: D1"), ModelResponse(text="RANKING: 1", model=MODEL)],
                        AgentConfig(final_order="forced_rerank"))
    assert plain.calls[0] == forced.calls[0]


# --- model searches --------------------------------------------------------------

def test_model_search_returns_10_at_800_characters_with_stable_handles() -> None:
    query = "kinase marker5 marker7"
    record, model = episode([respond(call("search", 1, query=query)),
                             finish(ranking_line(["D21", "D1"]))])
    hits = [d for d, s in INDEX.rank(query, 10) if s > 0]
    assert 0 < len(hits) <= 10
    content = tool_messages(model)[0]
    by_doc = {d: f"D{i}" for i, d in enumerate(BM25[:20], 1)}
    new = [d for d in hits if d not in by_doc]
    assert new and any(d in by_doc for d in hits)  # the query meets old and new documents
    for d in hits:
        if d in by_doc:
            assert f"[{by_doc[d]}] already shown above" in content
    for n, d in enumerate(new, 21):
        text = CORPUS[d]
        assert f"[D{n}] {text if len(text) <= 800 else text[:800].rstrip() + ' …'}" in content
    assert "Search 2 of 3" in content and "1 search(es) left." in content
    assert record.candidates == BM25[:20] + new
    assert record.searches[1]["source"] == "model" and record.searches[1]["new_handles"][0] == "D21"
    assert record.ranking[:2] == [new[0], BM25[0]]
    assert record.tool_calls["executed_searches"] == 1


def test_third_model_search_is_refused_without_running_bm25(monkeypatch) -> None:
    index = BM25Index(CORPUS)
    queries = []
    real = index.rank
    monkeypatch.setattr(index, "rank", lambda text, k=None: queries.append(text) or real(text, k))
    record, model = episode([respond(call("search", 1, query="kinase")), respond(call("search", 2, query="beta")),
                             respond(call("search", 3, query="gamma")), finish("RANKING: D1")], index=index)
    assert queries == [CLAIM, "kinase", "beta"]
    refused = tool_messages(model)[-1]
    assert refused.startswith("Error: no searches left")
    assert record.tool_calls["rejected"] == {"tool_refused": 1}
    assert record.tool_calls["executed_searches"] == 2 and record.model_calls == 4


def test_rejected_calls_use_no_search_budget() -> None:
    record, model = episode([
        respond(call("search", 1, query="  ")),
        respond(call("search", 2)),
        respond(call("lookup", 3, id="x")),
        respond(call("finish", 4)),  # invalid finish: no answer
        respond(call("search", 5, query="kinase"), call("search", 6, query="beta")),  # second: over limit
        respond(call("search", 7, query="gamma")),
        finish("RANKING: D1"),
    ])
    assert record.tool_calls["rejected"] == {"invalid_arguments": 2, "unknown_tool": 1, "invalid_finish": 1,
                                             "over_limit": 1}
    assert [s["query"] for s in record.searches] == [CLAIM, "kinase", "gamma"]
    assert record.tool_calls["requested"] == 8 and record.stop_reason == "finished"


def test_no_dataset_id_reaches_the_model() -> None:
    _, model = episode([respond(call("search", 1, query="kinase epsilon")), finish("RANKING: D2")],
                       AgentConfig(final_order="forced_rerank"))
    assert len(model.calls) == 3  # agent search, agent finish, rerank (script exhausted: S1 fallback)
    sent = json.dumps([[m.content for m in messages] for messages, _ in model.calls])
    assert not any(doc_id in sent for doc_id in CORPUS)


# --- stop reasons and fallbacks -----------------------------------------------------

def test_text_answer_without_tool_call_is_parsed() -> None:
    record, _ = episode([respond(text="Done.\nRANKING: D2, D1")])
    assert record.stop_reason == "no_tool_call"
    assert record.ranking[:2] == [BM25[1], BM25[0]] and record.agent_ranking_status == "repaired"


def test_max_model_calls_ends_with_first_retrieval_order() -> None:
    config = AgentConfig(max_model_calls=2, max_searches=5)
    record, _ = episode([respond(call("search", 1, query="kinase")), respond(call("search", 2, query="beta"))],
                        config)
    assert record.stop_reason == "max_steps" and record.model_calls == 2
    assert record.agent_ranking_status == "failed"
    assert record.ranking[:20] == BM25[:20] and record.ranking[: len(record.candidates)] == record.candidates


class RaisingModel:
    """Raises on the first call, then answers from a script (the rerank call)."""

    def __init__(self, later: list[ModelResponse]) -> None:
        self.script = ScriptedModel(later)
        self.calls = 0

    def complete(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("declined")
        return self.script.complete(messages, tools)


def test_model_error_falls_back_and_forced_rerank_still_runs() -> None:
    plain = run_episode(build_agent_graph(AgentConfig(), RaisingModel([]), INDEX, CORPUS), "q1", CLAIM)
    assert plain.stop_reason == "model_error" and plain.error == "RuntimeError: declined"
    assert plain.ranking == BM25 and plain.agent_ranking_status == "model_error"
    assert plain.totals["estimated_usd"] is None and plain.totals["calls_without_usage"] == 1

    model = RaisingModel([ModelResponse(text="RANKING: 2, 1", usage=Usage(10, 1), model=MODEL)])
    forced = run_episode(build_agent_graph(AgentConfig(final_order="forced_rerank"), model, INDEX, CORPUS), "q1",
                         CLAIM)
    assert forced.stop_reason == "model_error" and forced.rerank["status"] == "repaired"
    assert forced.ranking[:2] == [BM25[1], BM25[0]]


# --- repair rule ----------------------------------------------------------------------

def test_parse_handles_repair_rule() -> None:
    handles = {"D1": "a", "D2": "b", "D3": "c"}
    assert parse_handles("RANKING: D3, D1, D2", handles).status == "valid"
    p = parse_handles("first\nRANKING: D1\nRANKING: d3, D9, D3, x", handles)  # last line counts
    assert (p.order, p.status, p.unknown, p.duplicates, p.missing) == (["c", "a", "b"], "repaired", 1, 1, 2)
    assert parse_handles("no ranking here", handles).order == ["a", "b", "c"]
    assert parse_handles("RANKING: D7", handles).status == "failed"
    assert parse_handles(None, handles).status == "failed"


# --- accounting and records ----------------------------------------------------------

def test_tokens_and_cost_include_the_rerank_call() -> None:
    responses = [respond(call("search", 1, query="kinase"), usage=Usage(1000, 100)),
                 finish("RANKING: D1"), ModelResponse(text="RANKING: 1", usage=Usage(5000, 40), model=MODEL)]
    record, _ = episode(responses, AgentConfig(final_order="forced_rerank"))
    assert record.totals["calls"] == 3 and record.totals["input_tokens"] == 7000
    assert record.totals["output_tokens"] == 240
    expected = sum(estimate_cost(MODEL, u) for u in (Usage(1000, 100), Usage(1000, 100), Usage(5000, 40)))
    assert record.totals["estimated_usd"] == pytest.approx(expected)
    assert len(record.calls) == 2 and record.calls[0]["tool_calls"] == [{"name": "search",
                                                                         "arguments": {"query": "kinase"}}]
    json.dumps(asdict(record))  # serialisable as it is


def test_agent_module_never_touches_labels() -> None:
    tree = ast.parse(Path(agent_mod.__file__).read_text())
    imported = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    retrieval = {name.split(".")[3] for name in imported if name.startswith("oarp.envs.retrieval.")}
    assert retrieval <= {"bm25", "rerank"}  # never analysis, data (qrels) or metrics
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert not names & {"qrels", "judged", "relevant"}


def test_call_costs_price_each_call_and_mark_unknown_ones() -> None:
    responses = [respond(call("search", 1, query="kinase"), usage=Usage(1000, 100)), finish("RANKING: D1"),
                 ModelResponse(text="RANKING: 1", usage=Usage(5000, 40), model=MODEL)]
    record, _ = episode(responses, AgentConfig(final_order="forced_rerank"))
    costs = agent_mod.call_costs(record)
    assert costs["agent"] == [estimate_cost(MODEL, Usage(1000, 100))] * 2
    assert costs["rerank"] == [estimate_cost(MODEL, Usage(5000, 40))]
    failed = run_episode(build_agent_graph(AgentConfig(), RaisingModel([]), INDEX, CORPUS), "q1", CLAIM)
    assert agent_mod.call_costs(failed) == {"agent": [None], "rerank": []}
