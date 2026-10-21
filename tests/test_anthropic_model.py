"""Offline tests for the Anthropic adapter: a fake client records each request
and returns real SDK `Message` objects built from JSON. No network, no cost."""

from typing import Any

import pytest
from anthropic.types import Message as ApiMessage

from oarp.core.loop import run_agent
from oarp.core.tools import Tool, ToolRegistry, ToolSpec
from oarp.core.types import Message, StopReason, ToolCall
from oarp.providers.anthropic_model import (
    AnthropicModel,
    AnthropicSettings,
    ModelOutputTruncated,
    ModelRefused,
    build_request,
)

ADD = ToolSpec(
    "add",
    "Add two integers.",
    {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]},
)


def api_message(*content: dict, stop_reason: str = "tool_use", **extra: Any) -> ApiMessage:
    return ApiMessage.model_validate({
        "id": "msg_x", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
        "content": list(content), "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 20}, **extra,
    })


def tool_use(call_id: str, name: str, **args: Any) -> dict:
    return {"type": "tool_use", "id": call_id, "name": name, "input": args}


THINKING = {"type": "thinking", "thinking": "", "signature": "sig-abc"}


class FakeClient:
    def __init__(self, *responses: ApiMessage) -> None:
        self._responses = list(responses)
        self.requests: list[dict] = []
        self.messages = self  # so client.messages.create(...) works

    def create(self, **request: Any) -> ApiMessage:
        self.requests.append(request)
        return self._responses.pop(0)


def model(*responses: ApiMessage) -> tuple[AnthropicModel, FakeClient]:
    client = FakeClient(*responses)
    return AnthropicModel(client=client), client


# --- request translation -------------------------------------------------------------


def test_request_shape_and_generation_settings() -> None:
    request = build_request(
        AnthropicSettings(),
        [Message("system", "be careful"), Message("user", "add 1 and 2")],
        [ADD],
    )
    assert request["model"] == "claude-haiku-5-5"
    assert request["max_tokens"] == 4096
    assert request["system"] == "be careful"
    assert request["messages"] == [{"role": "user", "content": "add 1 and 2"}]
    assert request["tools"] == [{"name": "add", "description": "Add two integers.", "input_schema": ADD.parameters}]
    assert request["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert request["output_config"] == {"effort": "medium"}
    assert request["thinking"] == {"type": "adaptive", "display": "summarized"}
    for forbidden in ("temperature", "top_p", "top_k"):
        assert forbidden not in request


def test_no_tools_means_no_tool_choice() -> None:
    request = build_request(AnthropicSettings(effort=None, thinking_display=None), [Message("user", "hi")], [])
    for absent in ("tools", "tool_choice", "output_config", "thinking"):
        assert absent not in request


def test_tool_results_of_one_step_are_merged_into_one_user_message() -> None:
    history = [
        Message("user", "go"),
        Message("assistant", "", (ToolCall("c1", "add", {"a": 1, "b": 2}), ToolCall("c2", "add", {"a": 3, "b": 4}))),
        Message("tool", "3", tool_call_id="c1"),
        Message("tool", "Error: only 1 tool call(s) per step", tool_call_id="c2", is_error=True),
    ]
    messages = build_request(AnthropicSettings(), history, [ADD])["messages"]
    assert len(messages) == 3
    assert messages[2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "3", "is_error": False},
            {"type": "tool_result", "tool_use_id": "c2", "content": "Error: only 1 tool call(s) per step", "is_error": True},
        ],
    }


def test_history_without_raw_is_rebuilt_from_text_and_tool_calls() -> None:
    history = [Message("user", "go"), Message("assistant", "Adding.", (ToolCall("c1", "add", {"a": 1, "b": 2}),))]
    assert build_request(AnthropicSettings(), history, [ADD])["messages"][1]["content"] == [
        {"type": "text", "text": "Adding."},
        {"type": "tool_use", "id": "c1", "name": "add", "input": {"a": 1, "b": 2}},
    ]


def test_raw_from_another_provider_is_ignored() -> None:
    history = [Message("user", "go"), Message("assistant", "hi", raw={"provider": "other", "content": ["?"]})]
    assert build_request(AnthropicSettings(), history, [])["messages"][1]["content"] == [{"type": "text", "text": "hi"}]


def test_late_system_message_rejected() -> None:
    with pytest.raises(ValueError, match="only leading system messages"):
        build_request(AnthropicSettings(), [Message("user", "a"), Message("system", "b")], [])


# --- response parsing ----------------------------------------------------------------


def test_response_parsed_into_model_response() -> None:
    m, _ = model(api_message(THINKING, {"type": "text", "text": "Adding."}, tool_use("toolu_1", "add", a=1, b=2)))
    response = m.complete([Message("user", "go")], [ADD])
    assert response.text == "Adding."
    assert response.tool_calls == (ToolCall("toolu_1", "add", {"a": 1, "b": 2}),)
    assert (response.usage.input_tokens, response.usage.output_tokens) == (100, 20)
    assert (response.stop_reason, response.model) == ("tool_use", "claude-haiku-5-5")
    assert [b.type for b in response.raw["content"]] == ["thinking", "text", "tool_use"]


def test_max_tokens_raises_truncated() -> None:
    m, _ = model(api_message({"type": "text", "text": "cut o"}, stop_reason="max_tokens"))
    with pytest.raises(ModelOutputTruncated, match="max_tokens"):
        m.complete([Message("user", "go")], [])


def test_refusal_raises_with_category() -> None:
    refusal = api_message(stop_reason="refusal", stop_details={"type": "refusal", "category": "cyber", "explanation": ""})
    m, _ = model(refusal)
    with pytest.raises(ModelRefused, match="cyber"):
        m.complete([Message("user", "go")], [])


# --- through the real loop ----------------------------------------------------------------


def registry() -> ToolRegistry:
    return ToolRegistry([Tool(ADD, lambda a, b: str(a + b))])


def test_full_loop_replays_thinking_blocks_unchanged() -> None:
    first = api_message(THINKING, tool_use("toolu_1", "add", a=17, b=25))
    second = api_message(tool_use("toolu_2", "finish", answer="42"))
    m, client = model(first, second)
    result = run_agent(m, registry(), system_prompt="sys", task_prompt="17+25?", max_steps=5)

    assert result.stop_reason is StopReason.FINISHED and result.final_answer == "42"
    replayed = client.requests[1]["messages"][1]
    assert replayed["role"] == "assistant"
    assert replayed["content"] == first.content  # same blocks, thinking signature intact
    assert replayed["content"][0] is first.content[0]  # not even copied
    assert client.requests[1]["messages"][2]["content"][0] == {
        "type": "tool_result", "tool_use_id": "toolu_1", "content": "42", "is_error": False,
    }
    # The loop also sees `finish` as a tool offered to the provider.
    assert [t["name"] for t in client.requests[0]["tools"]] == ["add", "finish"]


def test_provider_problems_become_model_error_in_the_loop() -> None:
    m, _ = model(api_message({"type": "text", "text": "x"}, stop_reason="max_tokens"))
    result = run_agent(m, registry(), system_prompt="sys", task_prompt="go", max_steps=3)
    assert result.stop_reason is StopReason.MODEL_ERROR
    assert result.error.startswith("ModelOutputTruncated")


def test_thinking_summaries_become_reasoning_summary() -> None:
    summarized = {"type": "thinking", "thinking": "The range end is off by one.", "signature": "sig"}
    m, _ = model(api_message(summarized, tool_use("toolu_1", "add", a=1, b=2)))
    response = m.complete([Message("user", "go")], [ADD])
    assert response.reasoning_summary == "The range end is off by one."


def test_empty_thinking_gives_no_reasoning_summary() -> None:
    m, _ = model(api_message(THINKING, tool_use("toolu_1", "add", a=1, b=2)))
    assert m.complete([Message("user", "go")], [ADD]).reasoning_summary is None
