"""The model interface and a deterministic scripted model for tests."""

from collections.abc import Sequence
from typing import Protocol

from oarp.core.types import Message, ModelResponse, ToolSpec


class ModelClient(Protocol):
    """Anything that maps (conversation, available tools) to one response.

    Adapters must translate provider errors into exceptions; the loop turns an
    exception into the MODEL_ERROR stop reason. Retries, if any, belong in the
    adapter, not the loop.
    """

    def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> ModelResponse: ...


class ScriptExhausted(RuntimeError):
    pass


class ScriptedModel:
    """Returns pre-written responses in order, ignoring its inputs.

    It records what it was shown, so tests can assert on what the loop sent
    (e.g. that a tool error was fed back to the model).
    """

    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[list[Message], list[ToolSpec]]] = []

    def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> ModelResponse:
        self.calls.append((list(messages), list(tools)))
        if len(self.calls) > len(self._responses):
            raise ScriptExhausted(f"script has only {len(self._responses)} responses")
        return self._responses[len(self.calls) - 1]
