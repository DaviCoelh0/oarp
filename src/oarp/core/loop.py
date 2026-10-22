"""The bounded tool-calling loop.

One step = one model call, followed by handling the tool calls it requested.
The loop knows nothing about coding or retrieval: environments plug in through
the ToolRegistry, and success is judged afterwards by an environment-specific
verifier, never by the loop.

The step logic lives in small functions (`call_model`, `handle_tool_calls`,
`end_run`) shared by both loop implementations: `run_agent` below and the
LangGraph version in `oarp.core.graph_loop`. The implementations differ only
in control flow.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from oarp.core.model import ModelClient
from oarp.core.tools import ToolError, ToolRegistry, validate_arguments
from oarp.core.types import (
    ErrorKind,
    Message,
    RunResult,
    StepRecord,
    StopReason,
    ToolCall,
    ToolResult,
    ToolSpec,
)

FINISH_TOOL = ToolSpec(
    name="finish",
    description="Call this when you are done. Give your final answer or a short summary of what you did.",
    parameters={
        "type": "object",
        "properties": {"answer": {"type": "string", "description": "Final answer or summary."}},
        "required": ["answer"],
    },
)

Emit = Callable[..., None]  # emit(event_name, **fields)


def run_agent(
    model: ModelClient,
    tools: ToolRegistry,
    *,
    system_prompt: str,
    task_prompt: str,
    max_steps: int,
    max_tool_calls_per_step: int = 1,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> RunResult:
    """Run one episode.

    Budgets: `max_steps` bounds model calls. `max_tool_calls_per_step` (N)
    bounds executed tool calls per step, so executed calls <= max_steps * N.
    The two counts are different and are reported separately.

    Tool calls within one response are handled in order:
    - The first N calls (`finish` included) use the per-step slots: a regular
      call is executed; a valid `finish` stops the run; an invalid `finish`
      gets an error result and the run continues.
    - Calls beyond the first N are rejected: not executed, but each gets an
      error result so the model sees why (and every call stays paired with a
      result, as provider APIs require).
    - After a valid `finish`, remaining calls are ignored: no result is
      created and they stay visible only in the step's `response.tool_calls`.
      No further model call happens, so no pairing is needed.

    Observability: if `on_event` is given, the loop reports each model call,
    each tool result (with its outcome: executed / rejected / invalid_finish)
    and the run's end, as plain dicts, at the moment they happen. The loop
    knows nothing about where events go (files, JSON, ...).
    """
    specs = loop_tool_specs(tools, max_steps, max_tool_calls_per_step)
    emit = emitter(on_event)
    messages = [Message("system", system_prompt), Message("user", task_prompt)]
    steps: list[StepRecord] = []

    for step in range(1, max_steps + 1):
        record = call_model(model, messages, specs, step=step, emit=emit)
        if isinstance(record, str):
            return end_run(StopReason.MODEL_ERROR, steps, messages, emit, error=record)
        steps.append(record)
        messages.append(assistant_message(record))
        if not record.response.tool_calls:
            return end_run(StopReason.NO_TOOL_CALL, steps, messages, emit,
                           final_answer=record.response.text or None)

        phase = handle_tool_calls(record.response.tool_calls, tools, step=step,
                                  max_tool_calls_per_step=max_tool_calls_per_step, emit=emit)
        record.tool_results = phase.results
        messages.extend(phase.messages)
        if phase.finish_answer is not None:
            return end_run(StopReason.FINISHED, steps, messages, emit, final_answer=phase.finish_answer)

    return end_run(StopReason.MAX_STEPS, steps, messages, emit)


# --- Step logic shared by both loop implementations ---------------------------


def loop_tool_specs(tools: ToolRegistry, max_steps: int, max_tool_calls_per_step: int) -> list[ToolSpec]:
    """Check the budgets and tools, and return what the model is offered."""
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if max_tool_calls_per_step < 1:
        raise ValueError("max_tool_calls_per_step must be >= 1")
    if FINISH_TOOL.name in tools:
        raise ValueError(f"'{FINISH_TOOL.name}' is reserved by the loop")
    return [*tools.specs(), FINISH_TOOL]


def emitter(on_event: Callable[[dict[str, Any]], None] | None) -> Emit:
    def emit(event: str, **fields: Any) -> None:
        if on_event is not None:
            on_event({"event": event, **fields})
    return emit


def call_model(
    model: ModelClient, messages: list[Message], specs: list[ToolSpec], *, step: int, emit: Emit
) -> StepRecord | str:
    """One model call (one step). Returns the step's record (without tool
    results yet), or the error text if the model client raised."""
    started = time.perf_counter()
    try:
        response = model.complete(messages, specs)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        emit("model_error", step=step, latency_s=time.perf_counter() - started, error=error)
        return error
    record = StepRecord(step, response, time.perf_counter() - started)
    usage = response.usage
    emit("model_call", step=step, latency_s=record.model_latency_s, model=response.model,
         stop_reason=response.stop_reason,
         input_tokens=usage.input_tokens if usage else None,
         output_tokens=usage.output_tokens if usage else None,
         text=response.text, reasoning_summary=response.reasoning_summary,
         tool_calls=[{"id": c.id, "name": c.name, "arguments": c.arguments} for c in response.tool_calls])
    return record


def assistant_message(record: StepRecord) -> Message:
    response = record.response
    return Message("assistant", response.text, response.tool_calls, raw=response.raw)


@dataclass(frozen=True)
class ToolPhase:
    """The outcome of handling one response's tool calls."""

    results: list[ToolResult]  # executed and rejected calls, in order
    messages: list[Message]  # one tool message per result
    finish_answer: str | None = None  # set when a valid `finish` ended the run


def handle_tool_calls(
    calls: tuple[ToolCall, ...], tools: ToolRegistry, *, step: int, max_tool_calls_per_step: int, emit: Emit
) -> ToolPhase:
    """Apply the per-step limit and the `finish` rules (see `run_agent`) to one
    response's tool calls, executing the allowed ones. InfrastructureError
    from a tool propagates."""
    results: list[ToolResult] = []
    messages: list[Message] = []
    for index, call in enumerate(calls):
        tool_started = time.perf_counter()
        if index >= max_tool_calls_per_step:
            outcome = "rejected"
            tool_result = _error(
                call,
                ErrorKind.OVER_LIMIT,
                f"only {max_tool_calls_per_step} tool call(s) per step are allowed; "
                "this call was not executed",
            )
        elif call.name == FINISH_TOOL.name:
            try:
                validate_arguments(FINISH_TOOL.parameters, call.arguments)
            except ToolError as exc:
                outcome = "invalid_finish"
                tool_result = _error(call, ErrorKind.INVALID_FINISH, str(exc))
            else:
                return ToolPhase(results, messages, finish_answer=call.arguments["answer"])
        else:
            outcome = "executed"
            tool_result = tools.execute(call)
        results.append(tool_result)
        messages.append(
            Message("tool", tool_result.content, tool_call_id=call.id, is_error=tool_result.is_error)
        )
        emit("tool_result", step=step, call_id=call.id, name=call.name, arguments=call.arguments,
             outcome=outcome, is_error=tool_result.is_error, error_kind=tool_result.error_kind,
             latency_s=time.perf_counter() - tool_started, content=tool_result.content)
    return ToolPhase(results, messages)


def end_run(
    reason: StopReason, steps: list[StepRecord], messages: list[Message], emit: Emit,
    *, final_answer: str | None = None, error: str | None = None,
) -> RunResult:
    emit("run_end", stop_reason=reason, model_calls=len(steps), final_answer=final_answer, error=error)
    return RunResult(reason, steps, messages, final_answer=final_answer, error=error)


def _error(call: ToolCall, kind: ErrorKind, message: str) -> ToolResult:
    return ToolResult(call.id, call.name, f"Error: {message}", is_error=True, error_kind=kind)
