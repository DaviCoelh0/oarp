"""S3: a model-controlled retrieval agent, built as a LangGraph state graph.

The agent sees a claim and BM25's top 20 for it (exactly S1's information),
may search the corpus again with its own queries, and ends with one line

    RANKING: D7, D2, ...

naming documents by per-episode handles (D1, D2, ... in order of first
retrieval), never dataset ids. The graph:

    START -> forced_search -> model --(tool call)--> tools --(calls left)--> model
                                |                      | out of model calls
                                | finish / text only / model error
                                +-----------> [rerank] -> assemble -> END

- `forced_search` (harness): BM25 on the claim, top `first_k` at `first_chars`
  characters; handles D1..D20 in BM25 order; search 1 of `max_searches`.
- `model` (model behaviour): one agent model call, offered `search` and the
  loop's `finish`. The only node where the model decides anything.
- `tools` (harness): at most one tool call per step. A search runs BM25 on the
  model's query and shows the top `search_k` documents with a positive score at
  `search_chars` characters; a document seen before keeps its handle and is
  shown as one line. A search over budget is refused WITHOUT running BM25.
  Unknown tools, bad arguments and extra calls in one step are rejected with
  the core ErrorKinds; a rejected call uses no search budget.
- `rerank` (harness forces, model ranks; only for final_order="forced_rerank"):
  the S1 reranker on all candidates in first-retrieval order. With no extra
  search its request is S1's request for that claim, byte for byte.
- `assemble` (harness): the chosen order of the candidates (the agent's or the
  reranker's; candidates left out are appended in first-retrieval order), then
  BM25's ranking of the claim below the candidates, as S1 does below rank 20.

The agent's prompt does not depend on `final_order`, so the agent behaves the
same in both configurations and its ranking is recorded in both (paired
scoring). `build_agent_graph` is the only place where an `AgentConfig` becomes
structure; M3 can loop over configs. Nothing here reads relevance labels.

Requires the optional extras: pip install -e ".[retrieval,langgraph]".
"""

import hashlib
import json
import operator
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from oarp.core.loop import FINISH_TOOL, assistant_message, call_model, emitter
from oarp.core.model import ModelClient
from oarp.core.pricing import estimate_cost
from oarp.core.tools import ToolError, validate_arguments
from oarp.core.types import ErrorKind, Message, StepRecord, StopReason, ToolCall, ToolResult, ToolSpec, Usage
from oarp.envs.retrieval.bm25 import BM25Index
from oarp.envs.retrieval.rerank import RerankConfig, RerankRecord, rerank, truncate

AGENT_SYSTEM = (
    "You find and rank scientific abstracts that are relevant to a claim. An abstract is relevant if it "
    "contains evidence that supports or refutes the claim."
)
AGENT_INSTRUCTION = (
    "Relevant abstracts may be missing from these results. You may call the search tool with your own "
    "query up to {extra} more time(s); each search returns at most {search_k} abstracts, shortened. When you "
    "are done, call finish with exactly one line: RANKING: followed by all document handles (such as D3), "
    "most relevant first, separated by commas."
)
SEARCH_TOOL_DESCRIPTION = (
    "Search the corpus of scientific abstracts with BM25 keyword matching. Returns at most {search_k} "
    "abstracts, shortened, each with a handle such as D21. Abstracts shown before keep their handle."
)
_RANKING_LINE = re.compile(r"^\s*RANKING:\s*(.*)$", re.I | re.M)
_HANDLE = re.compile(r"\bD(\d+)\b", re.I)


@dataclass(frozen=True)
class AgentConfig:
    """Every harness setting of S3, numeric and structural; hashed and recorded."""

    first_k: int = 20  # forced first search: BM25 on the claim
    first_chars: int = 2000  # with first_k: exactly S1's information
    search_k: int = 10  # per model-requested search
    search_chars: int = 800
    max_searches: int = 3  # per episode, INCLUDING the forced first search
    max_model_calls: int = 8  # agent model calls; the rerank call is counted separately
    final_order: Literal["agent", "forced_rerank"] = "agent"
    rerank_doc_chars: int = 2000  # S1's confirmed setting; the reranker sees all candidates
    rerank_shuffle: bool = False

    def __post_init__(self) -> None:
        if self.final_order not in ("agent", "forced_rerank"):
            raise ValueError("final_order must be 'agent' or 'forced_rerank'")
        if min(self.first_k, self.first_chars, self.search_k, self.search_chars, self.max_searches,
               self.max_model_calls, self.rerank_doc_chars) < 1:
            raise ValueError("all numeric settings must be >= 1")


def config_sha256(config: AgentConfig) -> str:
    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()


def search_tool(config: AgentConfig) -> ToolSpec:
    return ToolSpec(
        name="search",
        description=SEARCH_TOOL_DESCRIPTION.format(search_k=config.search_k),
        parameters={"type": "object",
                    "properties": {"query": {"type": "string", "description": "Keywords to search for."}},
                    "required": ["query"]},
    )


def first_message(claim: str, texts: Sequence[str], config: AgentConfig) -> str:
    shown = "\n\n".join(f"[D{i}] {truncate(t, config.first_chars)}" for i, t in enumerate(texts, 1))
    instruction = AGENT_INSTRUCTION.format(extra=config.max_searches - 1, search_k=config.search_k)
    return f"Claim: {claim}\n\nA search for the claim returned these {len(texts)} abstracts:\n\n{shown}\n\n{instruction}"


@dataclass(frozen=True)
class SearchRecord:
    query: str
    source: Literal["harness", "model"]
    handles: list[str]  # returned, in BM25 order
    new_handles: list[str]  # first seen in this search


@dataclass(frozen=True)
class ParsedHandles:
    order: list[str]  # doc ids, best first: the named candidates, then the rest in first-retrieval order
    status: str  # valid | repaired | failed | model_error
    unknown: int = 0  # handles not in the episode's table
    duplicates: int = 0
    missing: int = 0


def parse_handles(text: str | None, handles: dict[str, str]) -> ParsedHandles:
    """The S1 repair rule on handles: the last RANKING line counts; unknown
    handles are dropped, repeats keep their first position, candidates left
    out are appended in first-retrieval order."""
    candidates = list(handles.values())
    lines = _RANKING_LINE.findall(text or "")
    named = [f"D{int(n)}" for n in _HANDLE.findall(lines[-1])] if lines else []
    order: list[str] = []
    unknown = duplicates = 0
    for handle in named:
        if handle not in handles:
            unknown += 1
        elif handles[handle] in order:
            duplicates += 1
        else:
            order.append(handles[handle])
    if not order:
        return ParsedHandles(candidates, "failed", unknown, duplicates, len(candidates))
    missing = [d for d in candidates if d not in order]
    status = "valid" if not (unknown or duplicates or missing) else "repaired"
    return ParsedHandles(order + missing, status, unknown, duplicates, len(missing))


class AgentState(TypedDict):
    query_id: str
    claim: str
    messages: Annotated[list[Message], operator.add]
    steps: Annotated[list[StepRecord], operator.add]  # completed agent steps
    pending: StepRecord | None  # the current step before its tool results exist
    model_calls: int  # agent model calls attempted (a failed call counts)
    searches: Annotated[list[SearchRecord], operator.add]
    handles: dict[str, str]  # handle -> doc id, in order of first retrieval
    bm25_ranking: list[str]  # BM25's full ranking of the claim
    stop_reason: StopReason | None
    final_answer: str | None
    error: str | None
    agent_ranking: ParsedHandles | None
    rerank_record: RerankRecord | None
    ranking: list[str] | None  # the final full ranking that is scored


@dataclass(frozen=True)
class AgentGraph:
    config: AgentConfig
    graph: Any  # the compiled LangGraph graph
    recursion_limit: int


@dataclass
class EpisodeRecord:
    """One episode, JSON-serialisable with `asdict`. Contains no labels."""

    query_id: str
    config: dict
    config_sha256: str
    stop_reason: str
    error: str | None
    model_calls: int
    tool_calls: dict[str, Any]  # requested / executed searches / finish / rejected by error kind
    searches: list[dict]
    candidates: list[str]  # doc ids in first-retrieval order
    agent_ranking: list[str]  # the agent's order of the candidates (repaired)
    agent_ranking_status: str
    agent_ranking_repairs: dict[str, int]
    rerank: dict | None  # the S1 rerank record when the rerank node ran
    ranking: list[str]  # final full ranking
    calls: list[dict]  # per agent call: tokens, latency, text, tool calls, ...
    totals: dict[str, Any] = field(default_factory=dict)  # agent + rerank tokens and estimated $


def build_agent_graph(config: AgentConfig, model: ModelClient, index: BM25Index,
                      corpus: dict[str, str]) -> AgentGraph:
    """Assemble and compile the graph for one config (once per config, then
    `run_episode` per claim). The model, index and corpus are runtime
    dependencies, not settings."""
    specs = [search_tool(config), FINISH_TOOL]
    emit = emitter(None)

    def forced_search(state: AgentState) -> dict[str, Any]:
        ranking = [d for d, _ in index.rank(state["claim"])]
        top = ranking[: config.first_k]
        handles = {f"D{i}": d for i, d in enumerate(top, 1)}
        return {
            "bm25_ranking": ranking, "handles": handles,
            "searches": [SearchRecord(state["claim"], "harness", list(handles), list(handles))],
            "messages": [Message("system", AGENT_SYSTEM),
                         Message("user", first_message(state["claim"], [corpus[d] for d in top], config))],
        }

    def model_node(state: AgentState) -> dict[str, Any]:
        step = state["model_calls"] + 1
        record = call_model(model, state["messages"], specs, step=step, emit=emit)
        if isinstance(record, str):  # the client raised: no step is recorded
            return {"model_calls": step, "stop_reason": StopReason.MODEL_ERROR, "error": record}
        update: dict[str, Any] = {"model_calls": step, "messages": [assistant_message(record)]}
        if not record.response.tool_calls:
            update.update(steps=[record], stop_reason=StopReason.NO_TOOL_CALL,
                          final_answer=record.response.text or None)
        else:
            update["pending"] = record
        return update

    def run_search(query: str, state: AgentState) -> tuple[SearchRecord, dict[str, str], str]:
        handles = dict(state["handles"])
        by_doc = {d: h for h, d in handles.items()}
        hits = [d for d, score in index.rank(query, config.search_k) if score > 0]
        returned, new, lines = [], [], []
        for doc_id in hits:
            if doc_id in by_doc:
                lines.append(f"[{by_doc[doc_id]}] already shown above")
            else:
                handle = f"D{len(handles) + 1}"
                handles[handle] = by_doc[doc_id] = doc_id
                new.append(handle)
                lines.append(f"[{handle}] {truncate(corpus[doc_id], config.search_chars)}")
            returned.append(by_doc[doc_id])
        done = len(state["searches"]) + 1
        body = "\n\n".join(lines) if lines else "No results: no abstract matches these keywords."
        content = (f"Search {done} of {config.max_searches} for: {query}\n\n{body}\n\n"
                   f"{config.max_searches - done} search(es) left.")
        return SearchRecord(query, "model", returned, new), handles, content

    def tools_node(state: AgentState) -> dict[str, Any]:
        record = state["pending"]
        assert record is not None  # the router reaches tools only after a response with tool calls
        results: list[ToolResult] = []
        update: dict[str, Any] = {"pending": None}
        for position, call in enumerate(record.response.tool_calls):
            if position >= 1:
                result = _error(call, ErrorKind.OVER_LIMIT, "only 1 tool call per step is allowed; "
                                "this call was not executed")
            elif call.name == FINISH_TOOL.name:
                try:
                    validate_arguments(FINISH_TOOL.parameters, call.arguments)
                except ToolError as exc:
                    result = _error(call, ErrorKind.INVALID_FINISH, str(exc))
                else:  # later calls in the same response are ignored, as in the core loop
                    update.update(stop_reason=StopReason.FINISHED, final_answer=call.arguments["answer"])
                    break
            elif call.name != "search":
                result = _error(call, ErrorKind.UNKNOWN_TOOL, f"unknown tool '{call.name}'")
            else:
                result = _search_call(call, state, update)
            results.append(result)
        update["steps"] = [replace(record, tool_results=results)]
        update["messages"] = [Message("tool", r.content, tool_call_id=r.call_id, is_error=r.is_error)
                              for r in results]
        if update.get("stop_reason") is None and state["model_calls"] >= config.max_model_calls:
            update["stop_reason"] = StopReason.MAX_STEPS
        return update

    def _search_call(call: ToolCall, state: AgentState, update: dict[str, Any]) -> ToolResult:
        try:
            validate_arguments(search_tool(config).parameters, call.arguments)
            if not call.arguments["query"].strip():
                raise ToolError("argument 'query' must not be empty")
        except ToolError as exc:
            return _error(call, ErrorKind.INVALID_ARGUMENTS, str(exc))
        if len(state["searches"]) >= config.max_searches:
            return _error(call, ErrorKind.TOOL_REFUSED,
                          f"no searches left (at most {config.max_searches} per episode, including the first "
                          "one); call finish with your ranking")
        search, handles, content = run_search(call.arguments["query"].strip(), state)
        update.update(searches=[search], handles=handles)
        return ToolResult(call.id, call.name, content)

    def rerank_node(state: AgentState) -> dict[str, Any]:
        candidates = list(state["handles"].values())
        settings = RerankConfig(k=len(candidates), max_doc_chars=config.rerank_doc_chars,
                                shuffle=config.rerank_shuffle)
        return {"rerank_record": rerank(model, state["query_id"], state["claim"], candidates, corpus, settings)}

    def assemble(state: AgentState) -> dict[str, Any]:
        if state["stop_reason"] == StopReason.MODEL_ERROR:
            parsed = replace(parse_handles(None, state["handles"]), status="model_error")
        else:
            parsed = parse_handles(state["final_answer"], state["handles"])
        order = state["rerank_record"].ranking if state["rerank_record"] is not None else parsed.order
        chosen = set(order)
        return {"agent_ranking": parsed, "ranking": order + [d for d in state["bm25_ranking"] if d not in chosen]}

    finalize = "rerank" if config.final_order == "forced_rerank" else "assemble"

    def after_model(state: AgentState) -> str:
        return finalize if state["stop_reason"] is not None else "tools"

    def after_tools(state: AgentState) -> str:
        return finalize if state["stop_reason"] is not None else "model"

    graph = StateGraph(AgentState)
    graph.add_node("forced_search", forced_search)
    graph.add_node("model", model_node)
    graph.add_node("tools", tools_node)
    graph.add_node("assemble", assemble)
    graph.add_edge(START, "forced_search")
    graph.add_edge("forced_search", "model")
    graph.add_conditional_edges("model", after_model, ["tools", finalize])
    graph.add_conditional_edges("tools", after_tools, ["model", finalize])
    if config.final_order == "forced_rerank":
        graph.add_node("rerank", rerank_node)
        graph.add_edge("rerank", "assemble")
    graph.add_edge("assemble", END)
    # Super-steps: forced_search, 2 per model call, rerank, assemble; +1 so a
    # correct run never reaches it (only a safety net against a routing bug).
    return AgentGraph(config, graph.compile(), 2 * config.max_model_calls + 4)


def run_episode(agent: AgentGraph, query_id: str, claim: str) -> EpisodeRecord:
    initial: AgentState = {
        "query_id": query_id, "claim": claim, "messages": [], "steps": [], "pending": None, "model_calls": 0,
        "searches": [], "handles": {}, "bm25_ranking": [], "stop_reason": None, "final_answer": None,
        "error": None, "agent_ranking": None, "rerank_record": None, "ranking": None,
    }
    final = agent.graph.invoke(initial, {"recursion_limit": agent.recursion_limit})
    return episode_record(agent.config, final)


def episode_record(config: AgentConfig, state: AgentState) -> EpisodeRecord:
    steps: list[StepRecord] = state["steps"]
    requested = sum(len(s.response.tool_calls) for s in steps)
    rejected: dict[str, int] = {}
    for s in steps:
        for r in s.tool_results:
            if r.is_error:
                rejected[str(r.error_kind)] = rejected.get(str(r.error_kind), 0) + 1
    model_searches = sum(1 for s in state["searches"] if s.source == "model")
    calls = [{
        "step": s.step, "model": s.response.model, "latency_s": s.model_latency_s,
        "input_tokens": s.response.usage.input_tokens if s.response.usage else None,
        "output_tokens": s.response.usage.output_tokens if s.response.usage else None,
        "stop_reason": s.response.stop_reason, "text": s.response.text,
        "reasoning_summary": s.response.reasoning_summary,
        "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in s.response.tool_calls],
    } for s in steps]
    rr = state["rerank_record"]
    usages = [(s.response.model, s.response.usage) for s in steps]
    if rr is not None:
        usages.append((rr.model, rr.usage))
    parsed = state["agent_ranking"]
    assert parsed is not None and state["ranking"] is not None and state["stop_reason"] is not None
    return EpisodeRecord(
        query_id=state["query_id"], config=asdict(config), config_sha256=config_sha256(config),
        stop_reason=str(state["stop_reason"]), error=state["error"], model_calls=state["model_calls"],
        tool_calls={"requested": requested, "executed_searches": model_searches,
                    "finish": int(state["stop_reason"] == StopReason.FINISHED), "rejected": rejected},
        searches=[asdict(s) for s in state["searches"]],
        candidates=list(state["handles"].values()),
        agent_ranking=parsed.order, agent_ranking_status=parsed.status,
        agent_ranking_repairs={"unknown": parsed.unknown, "duplicates": parsed.duplicates,
                               "missing": parsed.missing},
        rerank=None if rr is None else asdict(rr),
        ranking=state["ranking"], calls=calls, totals=_totals(usages, agent_failed_calls=_failed(state)),
    )


def _failed(state: AgentState) -> int:
    """Agent model calls that raised (they have no step record and no usage)."""
    return state["model_calls"] - len(state["steps"])


def _totals(usages: list[tuple[str | None, Any]], *, agent_failed_calls: int) -> dict[str, Any]:
    known = [(m, u) for m, u in usages if u is not None and u.input_tokens is not None
             and u.output_tokens is not None]
    costs = [estimate_cost(m, u) for m, u in known]
    complete = len(known) == len(usages) and agent_failed_calls == 0 and all(c is not None for c in costs)
    return {"calls": len(usages) + agent_failed_calls,
            "input_tokens": sum(u.input_tokens for _, u in known),
            "output_tokens": sum(u.output_tokens for _, u in known),
            "calls_without_usage": len(usages) - len(known) + agent_failed_calls,
            "estimated_usd": sum(costs) if complete else None}  # type: ignore[arg-type]


def call_costs(record: EpisodeRecord) -> dict[str, list[float | None]]:
    """Estimated $ per call: the agent calls (a call that raised, or one with
    unknown usage or model, is None) and the rerank call if it ran."""
    def cost(model: str | None, usage: Usage | None) -> float | None:
        return None if usage is None else estimate_cost(model, usage)

    agent = [cost(c["model"], None if c["input_tokens"] is None else Usage(c["input_tokens"], c["output_tokens"]))
             for c in record.calls]
    agent += [None] * (record.model_calls - len(record.calls))
    rr = record.rerank
    rerank_cost = [] if rr is None else [cost(rr["model"], None if rr["usage"] is None else Usage(**rr["usage"]))]
    return {"agent": agent, "rerank": rerank_cost}


def prompt_sha256() -> dict[str, str]:
    return {name: hashlib.sha256(text.encode()).hexdigest() for name, text in
            (("agent_system", AGENT_SYSTEM), ("agent_instruction", AGENT_INSTRUCTION),
             ("search_tool_description", SEARCH_TOOL_DESCRIPTION))}


def _error(call: ToolCall, kind: ErrorKind, message: str) -> ToolResult:
    return ToolResult(call.id, call.name, f"Error: {message}", is_error=True, error_kind=kind)
