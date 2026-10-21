"""The bounded tool-calling loop.

One step = one model call, followed by handling the tool calls it requested.
The loop knows nothing about coding or retrieval: environments plug in through
the ToolRegistry, and success is judged afterwards by an environment-specific
verifier, never by the loop.
"""

import time
from collections.abc import Callable
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
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if max_tool_calls_per_step < 1:
        raise ValueError("max_tool_calls_per_step must be >= 1")
    if FINISH_TOOL.name in tools:
        raise ValueError(f"'{FINISH_TOOL.name}' is reserved by the loop")

    specs = [*tools.specs(), FINISH_TOOL]
    messages = [Message("system", system_prompt), Message("user", task_prompt)]
    steps: list[StepRecord] = []

    def emit(event: str, **fields: Any) -> None:
        if on_event is not None:
            on_event({"event": event, **fields})

    def result(reason: StopReason, **kwargs: str | None) -> RunResult:
        emit("run_end", stop_reason=reason, model_calls=len(steps),
             final_answer=kwargs.get("final_answer"), error=kwargs.get("error"))
        return RunResult(reason, steps, messages, **kwargs)

    for step in range(1, max_steps + 1):
        started = time.perf_counter()
        try:
            response = model.complete(messages, specs)
        except Exception as exc:
            emit("model_error", step=step, latency_s=time.perf_counter() - started,
                 error=f"{type(exc).__name__}: {exc}")
            return result(StopReason.MODEL_ERROR, error=f"{type(exc).__name__}: {exc}")
        record = StepRecord(step, response, time.perf_counter() - started)
        steps.append(record)
        messages.append(Message("assistant", response.text, response.tool_calls, raw=response.raw))
        usage = response.usage
        emit("model_call", step=step, latency_s=record.model_latency_s, model=response.model,
             stop_reason=response.stop_reason,
             input_tokens=usage.input_tokens if usage else None,
             output_tokens=usage.output_tokens if usage else None,
             text=response.text, reasoning_summary=response.reasoning_summary,
             tool_calls=[{"id": c.id, "name": c.name, "arguments": c.arguments} for c in response.tool_calls])

        if not response.tool_calls:
            return result(StopReason.NO_TOOL_CALL, final_answer=response.text or None)

        for index, call in enumerate(response.tool_calls):
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
                    return result(StopReason.FINISHED, final_answer=call.arguments["answer"])
            else:
                outcome = "executed"
                tool_result = tools.execute(call)
            record.tool_results.append(tool_result)
            messages.append(
                Message("tool", tool_result.content, tool_call_id=call.id, is_error=tool_result.is_error)
            )
            emit("tool_result", step=step, call_id=call.id, name=call.name, arguments=call.arguments,
                 outcome=outcome, is_error=tool_result.is_error, error_kind=tool_result.error_kind,
                 latency_s=time.perf_counter() - tool_started, content=tool_result.content)

    return result(StopReason.MAX_STEPS)


def _error(call: ToolCall, kind: ErrorKind, message: str) -> ToolResult:
    return ToolResult(call.id, call.name, f"Error: {message}", is_error=True, error_kind=kind)
