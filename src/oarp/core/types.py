"""Provider-neutral data types shared by the loop, models and tools.

Every model adapter (scripted, Ollama, ...) converts its native format into
these types, so the loop never depends on a provider SDK.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolSpec:
    """What the model sees: a name, a description and a JSON Schema."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object: {"type": "object", ...}


@dataclass(frozen=True)
class ToolCall:
    """A model's request to run one tool."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """What the harness sends back to the model after a tool call."""

    call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class Message:
    """One conversation entry.

    Assistant messages may carry tool calls; tool messages carry exactly one
    result, linked to its call through `tool_call_id`.
    """

    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class ModelResponse:
    """One model turn: optional text plus zero or more tool calls."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage | None = None


class StopReason(StrEnum):
    """How a run terminated. This is NOT task success: a run that hit
    MAX_STEPS may still have solved the task, and a FINISHED run may not have.
    Success is decided afterwards by an independent, environment-specific
    verifier and recorded separately."""

    FINISHED = "finished"  # the model called the finish tool
    NO_TOOL_CALL = "no_tool_call"  # the model answered without calling a tool
    MAX_STEPS = "max_steps"  # step budget exhausted
    MODEL_ERROR = "model_error"  # the model client raised


@dataclass
class StepRecord:
    """Everything that happened in one step.

    One step = one model call. A step may execute several tool calls (up to the
    loop's per-step limit), so step counts and tool-call counts differ.
    `tool_results` holds results for executed and rejected calls; calls ignored
    after a valid `finish` appear only in `response.tool_calls`.
    """

    step: int
    response: ModelResponse
    model_latency_s: float
    tool_results: list[ToolResult] = field(default_factory=list)


@dataclass
class RunResult:
    """What the loop observed. Contains no task-success judgement (see StopReason)."""

    stop_reason: StopReason
    steps: list[StepRecord]
    messages: list[Message]
    final_answer: str | None = None
    error: str | None = None
