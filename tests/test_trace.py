import json
from pathlib import Path

import pytest

from oarp.core.pricing import estimate_cost
from oarp.core.trace import JsonlTracer, git_info, new_run_id, stable_hash
from oarp.core.types import StopReason, Usage

REPO = Path(__file__).resolve().parents[1]


def test_events_are_written_and_flushed_immediately(tmp_path: Path) -> None:
    path = tmp_path / "run" / "trace.jsonl"
    tracer = JsonlTracer(path)
    tracer.emit({"event": "run_start", "stop": StopReason.FINISHED})
    # Readable before close: a crash after this point still leaves the event on disk.
    line = json.loads(path.read_text().splitlines()[0])
    assert line["event"] == "run_start" and line["stop"] == "finished"
    assert "ts" in line and line["t_s"] >= 0
    tracer.close()


def test_model_calls_get_an_estimated_cost(tmp_path: Path) -> None:
    with JsonlTracer(tmp_path / "t.jsonl") as tracer:
        tracer.emit({"event": "model_call", "model": "claude-haiku-5-5", "input_tokens": 1000, "output_tokens": 200})
        tracer.emit({"event": "model_call", "model": "claude-haiku-5-5", "input_tokens": 1000, "output_tokens": 200})
    lines = [json.loads(l) for l in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert lines[0]["estimated_cost_usd"] == pytest.approx(1000 * 0.10e-6 + 200 * 0.50e-6)
    assert lines[0]["prices_as_of"] == "2026-10-19"
    assert tracer.total_cost_usd == pytest.approx(2 * (1000 * 0.10e-6 + 200 * 0.50e-6))
    assert tracer.cost_complete


def test_unknown_model_cost_is_none_not_zero(tmp_path: Path) -> None:
    with JsonlTracer(tmp_path / "t.jsonl") as tracer:
        tracer.emit({"event": "model_call", "model": None, "input_tokens": None, "output_tokens": None})
    assert not tracer.cost_complete


@pytest.mark.parametrize(
    ("model", "usage", "expected"),
    [
        ("claude-haiku-5-5", Usage(100_000, 1000), 100_000 * 0.10e-6 + 1000 * 0.50e-6),  # at the tier edge
        ("claude-haiku-5-5", Usage(100_001, 1000), 100_001 * 0.50e-6 + 1000 * 2.50e-6),  # long-prompt tier
        ("claude-sonnet-5-5", Usage(1000, 100), 1000 * 2e-6 + 100 * 10e-6),
        ("unknown-model", Usage(1, 1), None),
        ("claude-haiku-5-5", None, None),
    ],
)
def test_estimate_cost(model, usage, expected) -> None:
    assert estimate_cost(model, usage) == (pytest.approx(expected) if expected is not None else None)


def test_stable_hash_ignores_key_order() -> None:
    assert stable_hash({"a": 1, "b": [1, 2]}) == stable_hash({"b": [1, 2], "a": 1})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})


def test_run_ids_are_unique_and_sortable() -> None:
    ids = [new_run_id() for _ in range(50)]
    assert len(set(ids)) == 50
    assert all(len(i) == 22 and i[8] == "T" for i in ids)


def test_git_info(tmp_path: Path) -> None:
    info = git_info(REPO)
    assert info is not None and len(info["commit"]) == 40 and isinstance(info["dirty"], bool)
    assert git_info(tmp_path) is None  # not a repository
