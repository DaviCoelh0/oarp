"""Opt-in live test: real, PAID calls to Claude Haiku 5.5 (about $0.001).

Runs only with OARP_LIVE_TESTS=1; the key is read from oarp/.env and never printed:
    OARP_LIVE_TESTS=1 venv/bin/python -m pytest -m live -q -s

It uses the real tracer, so the run's exact tokens, estimated cost, latency and
any reasoning summaries are written to a trace (path printed with -s).
"""

import json
import os
import re
from pathlib import Path

import pytest

from oarp.core.loop import run_agent
from oarp.core.tools import Tool, ToolRegistry, ToolSpec
from oarp.core.trace import JsonlTracer
from oarp.core.types import StopReason

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("OARP_LIVE_TESTS") != "1", reason="set OARP_LIVE_TESTS=1 to run paid API tests"),
]

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def api_key() -> str:
    match = re.search(r'^\s*(?:export\s+)?ANTHROPIC_API_KEY\s*=\s*["\']?([^"\'\s]+)', ENV_FILE.read_text(), re.M)
    if not match:
        pytest.skip("ANTHROPIC_API_KEY not found in .env")
    return match.group(1)


def test_real_tool_loop_with_haiku(tmp_path_factory: pytest.TempPathFactory) -> None:
    from oarp.providers.anthropic_model import AnthropicModel, AnthropicSettings

    add = ToolSpec(
        "add",
        "Add two integers and return their sum.",
        {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]},
    )
    settings = AnthropicSettings(max_tokens=2048)
    assert settings.thinking_display == "summarized"  # the request shape under test
    model = AnthropicModel(settings, api_key=api_key())
    trace_path = tmp_path_factory.mktemp("live") / "trace.jsonl"

    with JsonlTracer(trace_path) as tracer:
        result = run_agent(
            model,
            ToolRegistry([Tool(add, lambda a, b: str(a + b))]),
            system_prompt="Use the add tool for additions. Always end by calling finish with the answer.",
            task_prompt="What is 17 + 25?",
            max_steps=4,
            on_event=tracer.emit,
        )

    calls = [e for e in map(json.loads, trace_path.read_text().splitlines()) if e["event"] == "model_call"]
    print(f"\ntrace: {trace_path}")
    for c in calls:
        print(f"  step {c['step']}: in={c['input_tokens']} out={c['output_tokens']} "
              f"latency={c['latency_s']:.2f}s cost=${c['estimated_cost_usd']} "
              f"reasoning_summary={'yes' if c['reasoning_summary'] else 'no'}"
              + (f" ({c['reasoning_summary'][:120]!r})" if c["reasoning_summary"] else ""))
    print(f"  total estimated cost: ${tracer.total_cost_usd:.6f}")

    # The requests (with adaptive thinking + summarized display) were accepted,
    # and any thinking blocks were replayed without a 400.
    assert result.stop_reason is StopReason.FINISHED, result.error
    assert "42" in (result.final_answer or "")
    assert all(s.response.model == "claude-haiku-5-5" for s in result.steps)
    assert all(c["input_tokens"] > 0 and c["estimated_cost_usd"] is not None for c in calls)
    # Summaries are reported, not required: adaptive thinking may skip thinking on a trivial task.
