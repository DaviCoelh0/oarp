"""S1 check-half runner and analyzer, free: scripted models, a fake Anthropic
client and the small BEIR-format fixture. No network, no paid call."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

pytest.importorskip("bm25s")

from oarp.core.types import ModelResponse, Usage  # noqa: E402
from oarp.envs.retrieval.data import dev_partition, load_scifact  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "retrieval"
HERE = REPO / "experiments" / "retrieval_s1_check"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"s1check_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load("run_check")
analyzer = _load("analyze")


class FixedModel:
    def __init__(self, answer=lambda k: "RANKING: " + ", ".join(map(str, range(1, k + 1))), usage=Usage(1000, 50)):
        self.answer, self.usage, self.calls = answer, usage, 0

    def complete(self, messages, tools) -> ModelResponse:
        self.calls += 1
        k = len(re.findall(r"^\[(\d+)\]", messages[-1].content, re.M))
        return ModelResponse(text=self.answer(k), usage=self.usage, model="fake", stop_reason="end_turn")


def run_fixture(model, out: Path, **kw) -> dict:
    return runner.run_check(model, data_dir=FIXTURE, out=out, paid=False, model_settings={"model": "fake"},
                            log=lambda _: None, **kw)


def test_schedule_shape() -> None:
    queries = [f"c{i}" for i in range(404)]
    schedule = runner.build_schedule(queries, queries[:20])
    assert len(schedule) == 444 and sum(e["pass"] == 1 for e in schedule) == 404
    assert {e["query_id"] for e in schedule if e["pass"] > 1} == set(queries[:20])
    assert {e["condition"] for e in schedule} == {"plain"}


def test_train_only_and_check_half_only(tmp_path: Path, monkeypatch) -> None:
    loaded = []

    def spy(d, split, **kw):
        loaded.append((split, kw))
        return load_scifact(d, split, **kw)

    monkeypatch.setattr(runner, "load_scifact", spy)
    monkeypatch.setattr(analyzer, "load_scifact", spy)
    manifest = run_fixture(FixedModel(), tmp_path)
    analyzer.analyze(tmp_path, FIXTURE)
    assert loaded == [("train", {}), ("train", {})]
    halves = dev_partition(sorted(load_scifact(FIXTURE, "train").queries))
    sample = json.loads((tmp_path / "private" / "sample.json").read_text())
    assert set(sample["queries"]) == set(halves["check"]) and not set(sample["queries"]) & set(halves["tune"])
    assert manifest["scheduled_calls"] == len(halves["check"]) + 2 * len(sample["stability"])


def test_refuses_when_prompts_or_config_differ_from_the_pilot(tmp_path: Path) -> None:
    good = {"prompt_sha256": runner.prompt_hashes(),
            "conditions": {"plain": {"rerank_config": {"k": 20, "max_doc_chars": 2000, "shuffle": False}}}}
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps(good))
    assert runner.check_same_as_pilot(path)["pilot_manifest_sha256"]
    path.write_text(json.dumps({**good, "prompt_sha256": {"rerank_system": "x", "rerank_instruction": "y"}}))
    with pytest.raises(SystemExit, match="prompts differ"):
        runner.check_same_as_pilot(path)
    bad = json.loads(json.dumps(good))
    bad["conditions"]["plain"]["rerank_config"]["k"] = 10
    path.write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="RerankConfig differs"):
        runner.check_same_as_pilot(path)
    with pytest.raises(SystemExit, match="not found"):
        runner.check_same_as_pilot(tmp_path / "missing.json")


def test_the_committed_pilot_manifest_matches_the_current_code() -> None:
    assert runner.check_same_as_pilot(runner.PILOT_MANIFEST)["pilot_commit"]


def test_dry_run_end_to_end_keeps_ids_private(tmp_path: Path) -> None:
    manifest = runner.main(["--dry-run", "--data-dir", str(FIXTURE), "--results", str(tmp_path)])
    assert manifest["mode"].startswith("dry-run") and manifest["completed_calls"] == manifest["scheduled_calls"]
    summary = analyzer.main(["--results", str(tmp_path), "--data-dir", str(FIXTURE)])
    assert summary["outcome_rules"]["run"] == "dry run: no outcome" and summary["outcome_rules"]["H1c"] is None
    public = "".join((tmp_path / n).read_text() for n in ("manifest.json", "summary.json", "analysis.md"))
    for qid in ("q1", "q2", "q3"):
        assert f'"{qid}"' not in public and f"{qid} " not in public


def test_copying_bm25_gives_zero_and_oracle_reaches_the_ceiling(tmp_path: Path) -> None:
    run_fixture(FixedModel(), tmp_path / "copy")
    summary, _ = analyzer.analyze(tmp_path / "copy", FIXTURE)
    assert summary["H1c_s1_minus_s0"]["mean"] == 0.0 and summary["harm_rate"] == 0.0
    assert summary["stability"]["queries_with_varying_ndcg"] == 0

    split = load_scifact(FIXTURE, "train")
    by_text = {text: doc for doc, text in split.corpus.items()}

    class OracleModel:
        def complete(self, messages, tools) -> ModelResponse:
            user = messages[-1].content
            qid = next(q for q, t in split.queries.items() if t == re.search(r"^Claim: (.*)$", user, re.M).group(1))
            shown = [by_text[t] for t in re.findall(r"^\[\d+\] (.*)$", user, re.M)]
            order = sorted(range(len(shown)), key=lambda i: -split.qrels[qid].get(shown[i], 0))
            return ModelResponse(text="RANKING: " + ", ".join(str(i + 1) for i in order), usage=Usage(1000, 50))

    run_fixture(OracleModel(), tmp_path / "oracle")
    summary, _ = analyzer.analyze(tmp_path / "oracle", FIXTURE)
    assert summary["means"]["s1_ndcg@10"] == pytest.approx(summary["means"]["oracle20_ndcg@10"])


def test_declines_are_counted_apart_from_other_errors(tmp_path: Path) -> None:
    class Declines:
        def complete(self, messages, tools):
            raise type("ModelRefused", (RuntimeError,), {})("request declined by the provider (category: bio)")

    run_fixture(Declines(), tmp_path)
    summary, _ = analyzer.analyze(tmp_path, FIXTURE)
    assert summary["declines"] == summary["calls"]["completed"] and summary["other_model_errors"] == 0
    assert summary["decline_categories"] == {"bio": summary["declines"]}
    assert summary["H1c_excluding_declined"] is None  # every query was declined


CI_POS, CI_ZERO, CI_NEG = ({"low": 0.01, "high": 0.1}, {"low": -0.01, "high": 0.1}, {"low": -0.1, "high": -0.01})


def rules(**overrides):
    base = dict(complete=True, paid=True, other_error_rate=0.0, decline_rate=0.0, h1=CI_POS, mean_cost_per_call=0.0013)
    return analyzer.classify(**{**base, **overrides})


def test_outcome_rules_at_their_thresholds() -> None:
    assert rules() == {"run": "valid", "H1c": "confirmed", "decline_warning": False, "cost_surprise": False}
    assert rules(h1=CI_ZERO)["H1c"] == "not confirmed" and rules(h1=CI_NEG)["H1c"] == "reversed"
    assert rules(other_error_rate=0.05)["run"] == "valid"
    assert rules(other_error_rate=0.051)["run"].startswith("invalid") and rules(other_error_rate=0.051)["H1c"] is None
    assert rules(decline_rate=0.10)["decline_warning"] is False and rules(decline_rate=0.101)["decline_warning"] is True
    assert rules(decline_rate=0.3)["run"] == "valid"  # declines alone never invalidate the run
    assert rules(mean_cost_per_call=0.0026)["cost_surprise"] is False
    assert rules(mean_cost_per_call=0.00261)["cost_surprise"] is True
    assert rules(complete=False)["run"].startswith("incomplete") and rules(paid=False)["H1c"] is None


# --- the real adapter path, with a fake client ---------------------------------------------

anthropic = pytest.importorskip("anthropic")
from anthropic.types import Message as ApiMessage  # noqa: E402

from oarp.providers import anthropic_model  # noqa: E402


class FakeClient:
    def __init__(self) -> None:
        self.requests = []
        self.messages = self

    def create(self, **request):
        self.requests.append(request)
        k = len(re.findall(r"^\[(\d+)\]", request["messages"][-1]["content"], re.M))
        return ApiMessage.model_validate({
            "id": "m", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
            "content": [{"type": "text", "text": "RANKING: " + ", ".join(map(str, range(1, k + 1)))}],
            "stop_reason": "end_turn", "stop_sequence": None, "usage": {"input_tokens": 9000, "output_tokens": 400}})


def test_paid_path_request_shape_and_no_key_leak(tmp_path: Path, monkeypatch, capsys) -> None:
    secret = "sk-ant-dummy-NEVER-WRITE-ME"
    (tmp_path / ".env").write_text(f"ANTHROPIC_API_KEY={secret}\n")
    clients = []
    real = anthropic_model.AnthropicModel

    def fake_model(settings, *, api_key=None, client=None):
        assert api_key == secret
        clients.append(FakeClient())
        return real(settings, client=clients[-1])

    monkeypatch.setattr(anthropic_model, "AnthropicModel", fake_model)
    monkeypatch.setattr(runner.study, "preflight", lambda *a, **kw: {"git": None})
    out = tmp_path / "out"
    manifest = runner.main(["--env-file", str(tmp_path / ".env"), "--data-dir", str(FIXTURE), "--results", str(out)])
    assert manifest["mode"] == "paid" and manifest["completed_calls"] == manifest["scheduled_calls"]
    request = clients[0].requests[0]
    assert "tools" not in request and request["output_config"] == {"effort": "medium"}
    assert request["max_tokens"] == 4096 and not {"temperature", "top_p", "top_k"} & set(request)
    written = "".join(p.read_text() for p in out.rglob("*") if p.is_file())
    assert secret not in written and secret not in capsys.readouterr().out
