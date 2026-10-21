"""The bounded tool-calling loop.

One step = one model call, followed by handling the tool calls it requested.
The loop knows nothing about coding or retrieval: environments plug in through
the ToolRegistry, and success is judged afterwards by an environment-specific
verifier, never by the loop.
"""

import time

from oarp.core.model import ModelClient
from oarp.core.tools import ToolError, ToolRegistry, validate_arguments
from oarp.core.types import (
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

    def result(reason: StopReason, **kwargs: str | None) -> RunResult:
        return RunResult(reason, steps, messages, **kwargs)

    for step in range(1, max_steps + 1):
        started = time.perf_counter()
        try:
            response = model.complete(messages, specs)
        except Exception as exc:
            return result(StopReason.MODEL_ERROR, error=f"{type(exc).__name__}: {exc}")
        record = StepRecord(step, response, time.perf_counter() - started)
        steps.append(record)
        messages.append(Message("assistant", response.text, response.tool_calls))

        if not response.tool_calls:
            return result(StopReason.NO_TOOL_CALL, final_answer=response.text or None)

        for index, call in enumerate(response.tool_calls):
            if index >= max_tool_calls_per_step:
                tool_result = _error(
                    call,
                    f"only {max_tool_calls_per_step} tool call(s) per step are allowed; "
                    "this call was not executed",
                )
            elif call.name == FINISH_TOOL.name:
                try:
                    validate_arguments(FINISH_TOOL.parameters, call.arguments)
                except ToolError as exc:
                    tool_result = _error(call, str(exc))
                else:
                    return result(StopReason.FINISHED, final_answer=call.arguments["answer"])
            else:
                tool_result = tools.execute(call)
            record.tool_results.append(tool_result)
            messages.append(Message("tool", tool_result.content, tool_call_id=call.id))

    return result(StopReason.MAX_STEPS)


def _error(call: ToolCall, message: str) -> ToolResult:
    return ToolResult(call.id, call.name, f"Error: {message}", is_error=True)
