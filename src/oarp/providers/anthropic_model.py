"""ModelClient adapter for the Anthropic Messages API (official `anthropic` SDK).

The only module that knows Anthropic's wire format. It translates oarp
messages and tool specs into a request, and the response back into a
ModelResponse; the loop never sees provider details.

Provider facts this relies on (verified 2026-10-19/21):
- Every tool_use in an assistant turn needs a tool_result in the next user
  turn; all results of one turn go in ONE user message.
- Assistant content (including thinking blocks) must be replayed unchanged.
- Claude Haiku 5.5 rejects non-default sampling parameters, so none are sent.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import anthropic

from oarp.core.types import Message, ModelResponse, ToolCall, ToolSpec, Usage

PROVIDER = "anthropic"


class ModelOutputTruncated(RuntimeError):
    """The response hit max_tokens; a tool call in it may be incomplete."""


class ModelRefused(RuntimeError):
    """The provider's safety classifier declined the request."""


@dataclass(frozen=True)
class AnthropicSettings:
    model: str = "claude-haiku-5-5"
    max_tokens: int = 4096
    effort: str | None = "medium"  # output_config.effort; None = provider default
    # Adaptive thinking with readable summaries. Display changes visibility only
    # (per the docs, thinking happens and is billed the same); keep it fixed
    # across compared conditions. None = omit `thinking` (provider default).
    thinking_display: str | None = "summarized"
    disable_parallel_tool_use: bool = True  # matches the loop's default of 1 call per step
    timeout_s: float = 60.0
    max_retries: int = 2  # SDK retries connection errors, 408/409/429/5xx


class AnthropicModel:
    def __init__(
        self,
        settings: AnthropicSettings = AnthropicSettings(),
        *,
        client: Any = None,
        api_key: str | None = None,
    ) -> None:
        self.settings = settings
        # `client` is injectable so tests can use a fake; otherwise the SDK
        # resolves credentials (api_key argument, then ANTHROPIC_API_KEY, ...).
        self._client = client or anthropic.Anthropic(
            api_key=api_key, timeout=settings.timeout_s, max_retries=settings.max_retries
        )

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelResponse:
        response = self._client.messages.create(**build_request(self.settings, messages, tools))
        return parse_response(response)


def build_request(
    settings: AnthropicSettings, messages: Sequence[Message], tools: Sequence[ToolSpec]
) -> dict[str, Any]:
    system: list[str] = []
    body: list[dict[str, Any]] = []
    for index, m in enumerate(messages):
        if m.role == "system":
            if body:
                raise ValueError(f"system message at position {index}: only leading system messages are supported")
            system.append(m.content)
        elif m.role == "user":
            body.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            body.append({"role": "assistant", "content": _assistant_content(m)})
        elif m.role == "tool":
            block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content, "is_error": m.is_error}
            previous = body[-1] if body else None
            if previous and previous["role"] == "user" and _is_tool_results(previous["content"]):
                previous["content"].append(block)  # one user turn for all results of a step
            else:
                body.append({"role": "user", "content": [block]})

    request: dict[str, Any] = {"model": settings.model, "max_tokens": settings.max_tokens, "messages": body}
    if system:
        request["system"] = "\n\n".join(system)
    if tools:
        request["tools"] = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
        ]
        request["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": settings.disable_parallel_tool_use}
    if settings.effort is not None:
        request["output_config"] = {"effort": settings.effort}
    if settings.thinking_display is not None:
        request["thinking"] = {"type": "adaptive", "display": settings.thinking_display}
    return request


def parse_response(response: Any) -> ModelResponse:
    if response.stop_reason == "max_tokens":
        raise ModelOutputTruncated(
            f"response hit max_tokens ({response.usage.output_tokens} output tokens)"
        )
    if response.stop_reason == "refusal":
        category = getattr(response.stop_details, "category", None)
        raise ModelRefused(f"request declined by the provider (category: {category})")

    blocks = list(response.content)
    summaries = [b.thinking for b in blocks if b.type == "thinking" and b.thinking]
    return ModelResponse(
        text="".join(b.text for b in blocks if b.type == "text"),
        tool_calls=tuple(ToolCall(b.id, b.name, b.input) for b in blocks if b.type == "tool_use"),
        usage=Usage(response.usage.input_tokens, response.usage.output_tokens),
        stop_reason=response.stop_reason,
        model=response.model,
        reasoning_summary="\n\n".join(summaries) or None,
        raw={"provider": PROVIDER, "content": blocks},
    )


def _assistant_content(m: Message) -> list[Any]:
    """Replay our own raw blocks unchanged; otherwise rebuild from text and tool calls
    (e.g. history produced by a scripted model or another provider)."""
    if isinstance(m.raw, dict) and m.raw.get("provider") == PROVIDER:
        return list(m.raw["content"])
    content: list[dict[str, Any]] = []
    if m.content:
        content.append({"type": "text", "text": m.content})
    content += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in m.tool_calls]
    if not content:
        raise ValueError("cannot send an empty assistant message")
    return content


def _is_tool_results(content: Any) -> bool:
    return isinstance(content, list) and all(b.get("type") == "tool_result" for b in content)
