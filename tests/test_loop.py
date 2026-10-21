import pytest

from oarp.core.loop import run_agent
from oarp.core.model import ScriptedModel
from oarp.core.tools import Tool, ToolRegistry, ToolSpec
from oarp.core.types import ModelResponse, StopReason, ToolCall, Usage

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


def call(name: str, call_id: str = "c1", **arguments: object) -> ModelResponse:
    return ModelResponse(tool_calls=(ToolCall(call_id, name, dict(arguments)),))


def run(model: ScriptedModel, max_steps: int = 5, per_step: int = 1, tools: list[Tool] | None = None):
    return run_agent(
        model,
        ToolRegistry(tools if tools is not None else [ADD]),
        system_prompt="sys",
        task_prompt="task",
        max_steps=max_steps,
        max_tool_calls_per_step=per_step,
    )


def calls(*items: tuple[str, dict]) -> ModelResponse:
    """One response containing several tool calls, with ids c0, c1, ..."""
    return ModelResponse(
        tool_calls=tuple(ToolCall(f"c{i}", name, args) for i, (name, args) in enumerate(items))
    )


def counting_add() -> tuple[Tool, list[tuple[int, int]]]:
    """An `add` tool that records every invocation of its handler."""
    invocations: list[tuple[int, int]] = []

    def handler(a: int, b: int) -> str:
        invocations.append((a, b))
        return str(a + b)

    return Tool(ADD.spec, handler), invocations


def test_tool_then_finish() -> None:
    model = ScriptedModel([call("add", a=2, b=3), call("finish", "c2", answer="5")])
    result = run(model)

    assert result.stop_reason is StopReason.FINISHED
    assert result.final_answer == "5"
    assert len(result.steps) == 2
    assert result.steps[0].tool_results[0].content == "5"
    # The second model call saw the tool result in its conversation.
    second_messages, specs = model.calls[1]
    assert second_messages[-1].role == "tool"
    assert second_messages[-1].content == "5"
    assert second_messages[-1].tool_call_id == "c1"
    assert [s.name for s in specs] == ["add", "finish"]


def test_max_steps_is_a_hard_budget() -> None:
    model = ScriptedModel([call("add", a=1, b=1)] * 10)
    result = run(model, max_steps=3)

    assert result.stop_reason is StopReason.MAX_STEPS
    assert len(model.calls) == 3
    assert result.final_answer is None


def test_plain_text_answer_stops_with_no_tool_call() -> None:
    result = run(ScriptedModel([ModelResponse(text="I give up")]))
    assert result.stop_reason is StopReason.NO_TOOL_CALL
    assert result.final_answer == "I give up"


def test_tool_error_is_fed_back_and_run_continues() -> None:
    model = ScriptedModel(
        [call("delete_everything"), call("add", a="x", b=1), call("finish", answer="done")]
    )
    result = run(model)

    assert result.stop_reason is StopReason.FINISHED
    errors = [step.tool_results[0] for step in result.steps[:2]]
    assert all(r.is_error for r in errors)
    assert "unknown tool" in errors[0].content
    assert "must be of type integer" in errors[1].content
    assert model.calls[2][0][-1].content == errors[1].content


def test_invalid_finish_does_not_stop() -> None:
    model = ScriptedModel([call("finish"), call("finish", answer="ok")])
    result = run(model)
    assert result.stop_reason is StopReason.FINISHED
    assert result.steps[0].tool_results[0].is_error


def test_calls_after_finish_in_same_response_are_ignored() -> None:
    response = ModelResponse(
        tool_calls=(ToolCall("c1", "finish", {"answer": "x"}), ToolCall("c2", "add", {"a": 1, "b": 2}))
    )
    result = run(ScriptedModel([response]))
    assert result.stop_reason is StopReason.FINISHED
    assert result.steps[0].tool_results == []


def test_model_exception_stops_with_model_error() -> None:
    result = run(ScriptedModel([]))  # exhausted script raises on the first call
    assert result.stop_reason is StopReason.MODEL_ERROR
    assert "ScriptExhausted" in (result.error or "")
    assert result.steps == []


def test_finish_name_is_reserved() -> None:
    finish = Tool(ToolSpec("finish", "", {"type": "object"}), lambda: "")
    with pytest.raises(ValueError, match="reserved"):
        run_agent(ScriptedModel([]), ToolRegistry([finish]), system_prompt="", task_prompt="", max_steps=1)


def test_scripted_runs_are_deterministic() -> None:
    script = [call("add", a=2, b=3), call("finish", "c2", answer="5")]
    a, b = run(ScriptedModel(script)), run(ScriptedModel(script))
    assert a.messages == b.messages
    assert a.stop_reason == b.stop_reason


# --- per-step tool-call limit (max_tool_calls_per_step) ---


def test_calls_within_limit_run_in_order() -> None:
    add, invocations = counting_add()
    response = calls(("add", {"a": 1, "b": 1}), ("add", {"a": 2, "b": 2}))
    result = run(ScriptedModel([response, call("finish", answer="ok")]), per_step=2, tools=[add])

    assert invocations == [(1, 1), (2, 2)]
    assert [r.content for r in result.steps[0].tool_results] == ["2", "4"]


def test_calls_beyond_limit_are_rejected_not_executed() -> None:
    add, invocations = counting_add()
    response = calls(("add", {"a": 1, "b": 1}), ("add", {"a": 2, "b": 2}))
    model = ScriptedModel([response, call("finish", answer="ok")])
    result = run(model, per_step=1, tools=[add])

    assert invocations == [(1, 1)]
    executed, rejected = result.steps[0].tool_results
    assert not executed.is_error
    assert rejected.is_error and rejected.call_id == "c1"
    assert "not executed" in rejected.content
    # Every requested call is paired with a tool message the model sees next.
    assert [m.tool_call_id for m in model.calls[1][0] if m.role == "tool"] == ["c0", "c1"]


def test_finish_beyond_limit_is_rejected_and_run_continues() -> None:
    response = calls(("add", {"a": 1, "b": 1}), ("finish", {"answer": "early"}))
    result = run(ScriptedModel([response, call("finish", answer="late")]), per_step=1)

    assert result.stop_reason is StopReason.FINISHED
    assert result.final_answer == "late"
    assert result.steps[0].tool_results[1].is_error


def test_finish_first_ignores_later_calls() -> None:
    add, invocations = counting_add()
    response = calls(("finish", {"answer": "done"}), ("add", {"a": 1, "b": 1}))
    result = run(ScriptedModel([response]), per_step=1, tools=[add])

    assert result.stop_reason is StopReason.FINISHED
    assert invocations == []
    assert result.steps[0].tool_results == []
    assert not any(m.role == "tool" for m in result.messages)


def test_finish_between_calls_within_limit() -> None:
    add, invocations = counting_add()
    response = calls(
        ("add", {"a": 1, "b": 1}), ("finish", {"answer": "2"}), ("add", {"a": 9, "b": 9})
    )
    result = run(ScriptedModel([response]), per_step=3, tools=[add])

    assert result.stop_reason is StopReason.FINISHED
    assert invocations == [(1, 1)]
    assert len(result.steps[0].tool_results) == 1  # only the executed call has a result
    assert len(result.steps[0].response.tool_calls) == 3  # all requests remain visible


def test_executed_calls_bounded_by_steps_times_limit() -> None:
    add, invocations = counting_add()
    greedy = calls(*[("add", {"a": i, "b": i}) for i in range(3)])
    result = run(ScriptedModel([greedy] * 5), max_steps=2, per_step=1, tools=[add])

    assert result.stop_reason is StopReason.MAX_STEPS
    assert len(result.steps) == 2  # model calls
    assert len(invocations) == 2  # executed tool calls = max_steps * 1
    assert sum(len(s.tool_results) for s in result.steps) == 6  # 2 executed + 4 rejected


# --- remaining gaps from the review ---


def test_last_step_tool_result_is_never_seen_by_model() -> None:
    model = ScriptedModel([call("add", a=1, b=2)])
    result = run(model, max_steps=1)

    assert result.stop_reason is StopReason.MAX_STEPS
    assert len(model.calls) == 1
    assert result.messages[-1].role == "tool" and result.messages[-1].content == "3"


@pytest.mark.parametrize(("max_steps", "per_step"), [(0, 1), (1, 0)])
def test_budgets_must_be_positive(max_steps: int, per_step: int) -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        run(ScriptedModel([]), max_steps=max_steps, per_step=per_step)


def test_empty_text_without_tool_call_gives_no_final_answer() -> None:
    result = run(ScriptedModel([ModelResponse(text="")]))
    assert result.stop_reason is StopReason.NO_TOOL_CALL
    assert result.final_answer is None


def test_usage_is_preserved_in_step_record() -> None:
    usage = Usage(input_tokens=10, output_tokens=3)
    response = ModelResponse(tool_calls=(ToolCall("c1", "finish", {"answer": "x"}),), usage=usage)
    result = run(ScriptedModel([response]))
    assert result.steps[0].response.usage == usage


def test_model_sees_a_snapshot_per_call() -> None:
    model = ScriptedModel([call("add", a=2, b=3), call("finish", "c2", answer="5")])
    run(model)
    assert [len(messages) for messages, _ in model.calls] == [2, 4]
