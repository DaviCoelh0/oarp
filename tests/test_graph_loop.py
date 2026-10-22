"""Equivalence of the LangGraph loop with the plain loop.

Each scripted episode runs through `run_agent` and `run_graph_agent`; the two
must give identical results, conversations, model inputs and events, ignoring
only timing fields. Deterministic and free. Skipped without the langgraph extra.
"""

from dataclasses import dataclass, replace
from typing import Any

import pytest

pytest.importorskip("langgraph")

from oarp.core.graph_loop import run_graph_agent  # noqa: E402
from oarp.core.loop import run_agent  # noqa: E402
from oarp.core.model import ScriptedModel  # noqa: E402
from oarp.core.tools import InfrastructureError, Tool, ToolRegistry, ToolSpec  # noqa: E402
from oarp.core.types import ErrorKind, ModelResponse, RunResult, StopReason, ToolCall, Usage  # noqa: E402

ADD = Tool(
    ToolSpec(
        "add",
        "Add two integers.",
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    ),
    lambda a, b: str(a + b),
)


def broken(**_: object) -> str:
    raise RuntimeError("bug in the tool")


def infra_down(**_: object) -> str:
    raise InfrastructureError("sandbox unavailable")


BROKEN = Tool(ToolSpec("broken", "Always crashes.", {"type": "object", "properties": {}}), broken)
INFRA = Tool(ToolSpec("infra", "Infrastructure fails.", {"type": "object", "properties": {}}), infra_down)


def calls(*items: tuple[str, dict]) -> ModelResponse:
    """One response containing several tool calls, with ids c0, c1, ..."""
    return ModelResponse(
        tool_calls=tuple(ToolCall(f"c{i}", name, args) for i, (name, args) in enumerate(items))
    )


ONE = calls(("add", {"a": 1, "b": 1}))
FINISH = calls(("finish", {"answer": "done"}))


@dataclass
class Episode:
    result: RunResult | None
    raised: BaseException | None
    events: list[dict[str, Any]]
    model_inputs: list  # what the scripted model was shown, call by call


def episode(loop, script, *, max_steps=5, per_step=1, tools=(ADD, BROKEN, INFRA)) -> Episode:
    model = ScriptedModel(script)
    events: list[dict[str, Any]] = []
    try:
        result = loop(model, ToolRegistry(tools), system_prompt="sys", task_prompt="task",
                      max_steps=max_steps, max_tool_calls_per_step=per_step, on_event=events.append)
        raised = None
    except InfrastructureError as exc:
        result, raised = None, exc
    return Episode(result, raised, events, model.calls)


def without_timing(ep: Episode) -> tuple:
    result = ep.result
    if result is not None:
        result = replace(result, steps=[replace(s, model_latency_s=0.0) for s in result.steps])
    events = [{k: v for k, v in e.items() if k != "latency_s"} for e in ep.events]
    raised = None if ep.raised is None else (type(ep.raised), str(ep.raised))
    return result, raised, events, ep.model_inputs


SCENARIOS = {
    "tool_then_finish": dict(script=[ONE, FINISH]),
    "over_limit_rejected": dict(script=[calls(("add", {"a": 1, "b": 1}), ("add", {"a": 2, "b": 2})), FINISH]),
    "finish_beyond_limit_rejected": dict(script=[calls(("add", {"a": 1, "b": 1}), ("finish", {"answer": "x"})),
                                                 FINISH]),
    "invalid_finish_continues": dict(script=[calls(("finish", {})), calls(("finish", {"answer": 3})), FINISH]),
    "calls_after_finish_ignored": dict(script=[calls(("finish", {"answer": "x"}), ("add", {"a": 1, "b": 1}))],
                                       per_step=3),
    "finish_between_calls": dict(script=[calls(("add", {"a": 1, "b": 1}), ("finish", {"answer": "x"}),
                                               ("add", {"a": 2, "b": 2}))], per_step=3),
    "mixed_limit_3": dict(script=[calls(*[("add", {"a": i, "b": i}) for i in range(5)]), FINISH], per_step=3),
    "tool_errors_fed_back": dict(script=[calls(("nope", {})), calls(("add", {"a": "x", "b": 1})),
                                         calls(("broken", {})), FINISH]),
    "no_tool_call_with_text": dict(script=[ONE, ModelResponse(text="the answer is 2")]),
    "no_tool_call_empty_text": dict(script=[ModelResponse()]),
    "max_steps": dict(script=[ONE] * 10, max_steps=3),
    "max_steps_one": dict(script=[ONE] * 10, max_steps=1),
    "finish_on_last_step": dict(script=[ONE, ONE, FINISH], max_steps=3),
    "model_error_first_call": dict(script=[]),
    "model_error_later": dict(script=[ONE, ONE]),
    "infrastructure_error": dict(script=[ONE, calls(("infra", {})), FINISH]),
    "usage_and_raw_preserved": dict(script=[
        ModelResponse(text="t", tool_calls=ONE.tool_calls, usage=Usage(10, 2), stop_reason="tool_use",
                      model="m", reasoning_summary="r", raw={"blocks": [1, 2]}),
        FINISH,
    ]),
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_graph_loop_matches_plain_loop(name: str) -> None:
    plain = episode(run_agent, **SCENARIOS[name])
    graph = episode(run_graph_agent, **SCENARIOS[name])
    assert without_timing(graph) == without_timing(plain)
    assert graph.events, "every scenario emits events"


@pytest.mark.parametrize(
    ("name", "stop", "steps", "error_kinds"),
    [
        ("tool_then_finish", StopReason.FINISHED, 2, [None]),
        ("over_limit_rejected", StopReason.FINISHED, 2, [None, ErrorKind.OVER_LIMIT]),
        ("invalid_finish_continues", StopReason.FINISHED, 3, [ErrorKind.INVALID_FINISH] * 2),
        ("calls_after_finish_ignored", StopReason.FINISHED, 1, []),
        ("tool_errors_fed_back", StopReason.FINISHED, 4,
         [ErrorKind.UNKNOWN_TOOL, ErrorKind.INVALID_ARGUMENTS, ErrorKind.TOOL_BUG]),
        ("no_tool_call_with_text", StopReason.NO_TOOL_CALL, 2, [None]),
        ("max_steps", StopReason.MAX_STEPS, 3, [None] * 3),
        ("model_error_later", StopReason.MODEL_ERROR, 2, [None] * 2),
    ],
)
def test_graph_loop_outcomes(name: str, stop: StopReason, steps: int, error_kinds: list) -> None:
    """Pins the expected outcomes, so equivalence cannot hold by both loops being wrong the same way."""
    result = episode(run_graph_agent, **SCENARIOS[name]).result
    assert result is not None and result.stop_reason is stop
    assert len(result.steps) == steps
    assert [r.error_kind for s in result.steps for r in s.tool_results] == error_kinds


def test_model_error_records_no_step_and_counts_calls() -> None:
    ep = episode(run_graph_agent, [ONE, ONE])
    assert ep.result is not None and ep.result.error.startswith("ScriptExhausted")
    assert [e["event"] for e in ep.events][-2:] == ["model_error", "run_end"]
    assert ep.events[-2]["step"] == 3 and ep.events[-1]["model_calls"] == 2


def test_infrastructure_error_propagates_unwrapped() -> None:
    ep = episode(run_graph_agent, **SCENARIOS["infrastructure_error"])
    assert type(ep.raised) is InfrastructureError and ep.result is None
    assert "run_end" not in [e["event"] for e in ep.events]  # the run aborts; the runner marks it invalid


def test_step_budget_is_not_capped_by_recursion_limit() -> None:
    # LangGraph's default recursion_limit (25 super-steps) would stop this run at
    # step 13; our limit scales with max_steps, so only max_steps decides.
    plain = episode(run_agent, [ONE] * 40, max_steps=40)
    graph = episode(run_graph_agent, [ONE] * 40, max_steps=40)
    assert graph.result is not None and graph.result.stop_reason is StopReason.MAX_STEPS
    assert len(graph.model_inputs) == 40
    assert without_timing(graph) == without_timing(plain)


@pytest.mark.parametrize(("max_steps", "per_step"), [(0, 1), (1, 0)])
def test_budgets_must_be_positive(max_steps: int, per_step: int) -> None:
    with pytest.raises(ValueError):
        episode(run_graph_agent, [FINISH], max_steps=max_steps, per_step=per_step)


def test_finish_name_is_reserved() -> None:
    finish = Tool(ToolSpec("finish", "", {"type": "object", "properties": {}}), lambda: "")
    with pytest.raises(ValueError, match="reserved"):
        episode(run_graph_agent, [FINISH], tools=(finish,))


def test_no_callback_changes_nothing() -> None:
    script = [ONE, FINISH]
    traced = episode(run_graph_agent, script).result
    silent = run_graph_agent(ScriptedModel(script), ToolRegistry([ADD]), system_prompt="sys",
                             task_prompt="task", max_steps=5)
    assert traced is not None and silent.messages == traced.messages
