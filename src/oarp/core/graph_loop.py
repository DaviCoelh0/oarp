"""The bounded tool-calling loop, expressed as a LangGraph state graph.

Same contract as `oarp.core.loop.run_agent` (same arguments, same RunResult,
same events); only the control flow is represented differently:

    START -> model --(stop_reason set?)--> END
               ^           |no
               |           v
               +------- tools --(stop_reason set?)--> END

- `model` node: one model call (one step). Sets NO_TOOL_CALL or MODEL_ERROR.
- `tools` node: handles that step's tool calls. Sets FINISHED, or MAX_STEPS
  once the step budget (counted in model calls) is used up.
- One router after both nodes: END when a stop reason is set, else the other node.

The nodes are thin adapters between the graph state and the step functions in
`oarp.core.loop` (`call_model`, `handle_tool_calls`, `end_run`), which
`run_agent` uses too, so the step rules exist once. This module owns only the
state, the reducers and the routing. Its `recursion_limit` (which counts graph
super-steps, not model calls) is set above the largest possible number of
super-steps, so it is only a safety net against a routing bug.

Requires the optional extra: pip install -e ".[langgraph]".
"""

import operator
from collections.abc import Callable
from dataclasses import replace
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from oarp.core.loop import (
    assistant_message,
    call_model,
    emitter,
    end_run,
    handle_tool_calls,
    loop_tool_specs,
)
from oarp.core.model import ModelClient
from oarp.core.tools import ToolRegistry
from oarp.core.types import Message, RunResult, StepRecord, StopReason


class LoopState(TypedDict):
    # Annotated reducers say how a node's update is merged into the state:
    # messages and steps are appended; the other keys are simply overwritten.
    messages: Annotated[list[Message], operator.add]
    steps: Annotated[list[StepRecord], operator.add]  # completed steps only
    # The current step's record, from the model node, before its tool results
    # exist. The tools node completes it and appends it to `steps`.
    pending: StepRecord | None
    step: int  # model calls attempted so far (a failed call counts)
    stop_reason: StopReason | None
    final_answer: str | None
    error: str | None


def run_graph_agent(
    model: ModelClient,
    tools: ToolRegistry,
    *,
    system_prompt: str,
    task_prompt: str,
    max_steps: int,
    max_tool_calls_per_step: int = 1,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> RunResult:
    """Run one episode. Drop-in replacement for `run_agent`; see its docstring
    for the budget, tool-call and event semantics, which are identical."""
    specs = loop_tool_specs(tools, max_steps, max_tool_calls_per_step)
    emit = emitter(on_event)

    def model_node(state: LoopState) -> dict[str, Any]:
        step = state["step"] + 1
        record = call_model(model, state["messages"], specs, step=step, emit=emit)
        if isinstance(record, str):  # the model client raised: no step is recorded
            return {"step": step, "stop_reason": StopReason.MODEL_ERROR, "error": record}
        update: dict[str, Any] = {"step": step, "messages": [assistant_message(record)]}
        if not record.response.tool_calls:
            # The tools node will not run, so this step is already complete.
            update.update(steps=[record], stop_reason=StopReason.NO_TOOL_CALL,
                          final_answer=record.response.text or None)
        else:
            update["pending"] = record
        return update

    def tools_node(state: LoopState) -> dict[str, Any]:
        record = state["pending"]
        assert record is not None  # the router only reaches tools after a response with tool calls
        phase = handle_tool_calls(record.response.tool_calls, tools, step=record.step,
                                  max_tool_calls_per_step=max_tool_calls_per_step, emit=emit)
        update: dict[str, Any] = {
            "steps": [replace(record, tool_results=phase.results)],
            "pending": None,
            "messages": phase.messages,
        }
        if phase.finish_answer is not None:
            update.update(stop_reason=StopReason.FINISHED, final_answer=phase.finish_answer)
        elif state["step"] >= max_steps:
            update["stop_reason"] = StopReason.MAX_STEPS
        return update

    def route(next_node: str) -> Callable[[LoopState], str]:
        def router(state: LoopState) -> str:
            return END if state["stop_reason"] is not None else next_node
        return router

    graph = StateGraph(LoopState)
    graph.add_node("model", model_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route("tools"), ["tools", END])
    graph.add_conditional_edges("tools", route("model"), ["model", END])

    initial: LoopState = {
        "messages": [Message("system", system_prompt), Message("user", task_prompt)],
        "steps": [], "pending": None, "step": 0, "stop_reason": None, "final_answer": None, "error": None,
    }
    # At most 2 super-steps (model, tools) per model call; +1 so a correct run
    # can never reach the limit. Exceeding it raises GraphRecursionError.
    final = graph.compile().invoke(initial, {"recursion_limit": 2 * max_steps + 1})

    return end_run(final["stop_reason"], final["steps"], final["messages"], emit,
                   final_answer=final["final_answer"], error=final["error"])
