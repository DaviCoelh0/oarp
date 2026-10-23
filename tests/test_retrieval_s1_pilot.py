"""S1 pilot runner and analyzer, free: scripted models, a fake Anthropic client
and the small BEIR-format fixture. No network, no paid call."""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("bm25s")

from oarp.core.types import Message, ModelResponse, Usage  # noqa: E402
from oarp.envs.retrieval import rerank  # noqa: E402
from oarp.envs.retrieval.data import SCIFACT_SHA256, load_scifact  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "retrieval"
PILOT = REPO / "experiments" / "retrieval_s1_pilot"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"s1_{name}", PILOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load("run_pilot")
analyzer = _load("analyze")


class FixedModel:
    """Answers every call with the same usage; `answer(k)` makes the RANKING text."""

    def __init__(self, usage: Usage | None, answer=lambda k: "RANKING: " + ", ".join(map(str, range(1, k + 1)))):
        self.usage, self.answer, self.calls = usage, answer, 0

    def complete(self, messages, tools) -> ModelResponse:
        self.calls += 1
        k = len(re.findall(r"^\[(\d+)\]", messages[-1].content, re.M))
        return ModelResponse(text=self.answer(k), usage=self.usage, model="fake", stop_reason="end_turn")


def run_fixture(model, tmp_path: Path, **kw) -> dict:
    return runner.run_pilot(model, data_dir=FIXTURE, out=tmp_path, paid=False, model_settings={"model": "fake"},
                            log=lambda _: None, **kw)


# --- sample and schedule ---------------------------------------------------------

def test_sample_is_deterministic_and_label_free() -> None:
    ids = [f"t{i}" for i in range(300)]
    sample, subset = runner.select_sample(ids)
    assert len(sample) == 100 and subset == sample[:20]
    assert runner.select_sample(list(reversed(ids))) == (sample, subset)


def test_schedule_shape_and_condition_order() -> None:
    sample = [f"t{i}" for i in range(100)]
    schedule = runner.build_schedule(sample, sample[:20])
    assert len(schedule) == 280 and sum(e["pass"] == 1 for e in schedule) == 200
    assert [e["index"] for e in schedule] == list(range(1, 281))
    for i in range(0, 280, 2):
        a, b = schedule[i], schedule[i + 1]
        assert a["query_id"] == b["query_id"] and a["pass"] == b["pass"]
        assert {a["condition"], b["condition"]} == {"plain", "shuffled"}
        assert a["condition"] == runner.first_condition(a["query_id"])
    assert {e["query_id"] for e in schedule if e["pass"] > 1} == set(sample[:20])
    firsts = {runner.first_condition(q) for q in sample}
    assert firsts == {"plain", "shuffled"}  # both orders occur


def test_runner_and_analyzer_load_train_only(tmp_path: Path, monkeypatch) -> None:
    loaded = []

    def spy(d, split, **kw):
        loaded.append((split, kw))
        return load_scifact(d, split, **kw)

    monkeypatch.setattr(runner, "load_scifact", spy)
    monkeypatch.setattr(analyzer, "load_scifact", spy)
    run_fixture(FixedModel(Usage(1000, 50)), tmp_path)
    analyzer.analyze(tmp_path, FIXTURE)
    assert loaded == [("train", {}), ("train", {})]
    sample = json.loads((tmp_path / "private" / "sample.json").read_text())
    tune = runner.dev_partition(sorted(load_scifact(FIXTURE, "train").queries))["tune"]
    assert set(sample["sample"]) <= set(tune)


# --- budget guard ------------------------------------------------------------------

def test_budget_guard_stops_before_the_cap(tmp_path: Path) -> None:
    model = FixedModel(Usage(10_000, 4_000))  # $0.001 + $0.002 = $0.003 per call, below the $0.0035 bound
    manifest = run_fixture(model, tmp_path, budget_usd=0.01)
    assert model.calls == 3 and manifest["stopped_by_budget"]
    assert manifest["estimated_spend_usd"] == pytest.approx(0.009) and manifest["estimated_spend_usd"] <= 0.01


def test_missing_usage_counts_at_worst_case(tmp_path: Path) -> None:
    model = FixedModel(None)
    manifest = run_fixture(model, tmp_path, budget_usd=0.01)
    assert model.calls == 2 and manifest["estimated_spend_usd"] == pytest.approx(0.007)


def test_a_call_above_the_worst_case_stops_the_run(tmp_path: Path) -> None:
    model = FixedModel(Usage(50_000, 4_000))  # $0.007 > $0.0035
    manifest = run_fixture(model, tmp_path)
    assert model.calls == 1 and manifest["calls_above_worst_case"] == 1


# --- preflight ---------------------------------------------------------------------

def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args],
                   check=True, capture_output=True)


@pytest.fixture
def clean_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.py").write_text("x = 1\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


@pytest.fixture
def good_data(tmp_path: Path) -> Path:
    (tmp_path / "data" / "scifact").mkdir(parents=True)
    (tmp_path / "data" / "scifact" / "ARCHIVE.json").write_text(json.dumps({"sha256": SCIFACT_SHA256}))
    return tmp_path / "data"


def test_preflight_accepts_a_clean_state(clean_repo: Path, good_data: Path, tmp_path: Path) -> None:
    checks = runner.preflight(good_data, tmp_path / "out", paid=True, repo=clean_repo)
    assert checks["git"]["dirty"] is False and set(checks["prompt_sha256"]) == {"rerank_system", "rerank_instruction"}


def test_preflight_refuses_dirty_untracked_nonempty_or_wrong_data(clean_repo: Path, good_data: Path,
                                                                    tmp_path: Path) -> None:
    out = tmp_path / "out"
    (clean_repo / "src" / "a.py").write_text("x = 2\n")
    with pytest.raises(SystemExit, match="uncommitted"):
        runner.preflight(good_data, out, paid=True, repo=clean_repo)
    _git(clean_repo, "commit", "-q", "-am", "edit")
    (clean_repo / "src" / "new.py").write_text("")
    with pytest.raises(SystemExit, match="untracked"):
        runner.preflight(good_data, out, paid=True, repo=clean_repo)
    (clean_repo / "src" / "new.py").unlink()
    out.mkdir()
    (out / "manifest.json").write_text("{}")
    with pytest.raises(SystemExit, match="not empty"):
        runner.preflight(good_data, out, paid=True, repo=clean_repo)
    (out / "manifest.json").unlink()
    (good_data / "scifact" / "ARCHIVE.json").write_text(json.dumps({"sha256": "0" * 64}))
    with pytest.raises(SystemExit, match="SHA-256"):
        runner.preflight(good_data, out, paid=True, repo=clean_repo)


def test_missing_key_refuses_before_any_call(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("OTHER=1\n")
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        runner.main(["--env-file", str(tmp_path / ".env"), "--data-dir", str(FIXTURE), "--results", str(tmp_path / "o")])


# --- the real adapter path, with a fake client --------------------------------------

anthropic = pytest.importorskip("anthropic")
from anthropic.types import Message as ApiMessage  # noqa: E402

from oarp.providers import anthropic_model  # noqa: E402


def api_text(text: str, stop_reason: str = "end_turn") -> ApiMessage:
    return ApiMessage.model_validate({
        "id": "msg_x", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
        "content": [{"type": "text", "text": text}], "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 7000, "output_tokens": 300}})


class FakeClient:
    def __init__(self, make) -> None:
        self.make, self.requests = make, []
        self.messages = self

    def create(self, **request):
        self.requests.append(request)
        k = len(re.findall(r"^\[(\d+)\]", request["messages"][-1]["content"], re.M))
        return self.make(k)


def test_paid_path_request_shape_and_no_key_leak(tmp_path: Path, monkeypatch, capsys) -> None:
    secret = "sk-ant-dummy-NEVER-WRITE-ME"
    (tmp_path / ".env").write_text(f"ANTHROPIC_API_KEY={secret}\n")
    clients = []
    real = anthropic_model.AnthropicModel

    def fake_model(settings, *, api_key=None, client=None):
        assert api_key == secret  # the key reaches the adapter, and only the adapter
        fake = FakeClient(lambda k: api_text("RANKING: " + ", ".join(map(str, range(1, k + 1)))))
        clients.append(fake)
        return real(settings, client=fake)

    monkeypatch.setattr(anthropic_model, "AnthropicModel", fake_model)
    monkeypatch.setattr(runner, "preflight", lambda *a, **kw: {"git": None, "script_sha256": {}, "prompt_sha256": {}})
    out = tmp_path / "out"
    manifest = runner.main(["--env-file", str(tmp_path / ".env"), "--data-dir", str(FIXTURE), "--results", str(out)])
    assert manifest["mode"] == "paid" and manifest["completed_calls"] == manifest["scheduled_calls"] == 12
    request = clients[0].requests[0]
    assert "tools" not in request and "tool_choice" not in request
    assert request["model"] == "claude-haiku-5-5" and request["max_tokens"] == 4096
    assert request["output_config"] == {"effort": "medium"}
    assert request["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert not {"temperature", "top_p", "top_k"} & set(request)
    written = "".join(p.read_text() for p in out.rglob("*") if p.is_file())
    assert secret not in written and secret not in capsys.readouterr().out
    assert manifest["estimated_spend_usd"] == pytest.approx(12 * (7000 * 0.10 + 300 * 0.50) / 1e6)


def test_truncation_becomes_a_counted_model_error() -> None:
    model = anthropic_model.AnthropicModel(client=FakeClient(lambda k: api_text("RANKING: 1", "max_tokens")))
    record = rerank.rerank(model, "q1", "claim", ["d1", "d2"], {"d1": "a", "d2": "b"})
    assert record.status == "model_error" and record.error_kind == "ModelOutputTruncated"
    assert record.ranking == ["d1", "d2"]


# --- dry run end to end, analyzer, outcome rules -----------------------------------------------------

def test_dry_run_end_to_end_keeps_ids_private(tmp_path: Path) -> None:
    manifest = runner.main(["--dry-run", "--data-dir", str(FIXTURE), "--results", str(tmp_path)])
    assert manifest["mode"].startswith("dry-run") and manifest["completed_calls"] == 12
    summary = analyzer.main(["--results", str(tmp_path), "--data-dir", str(FIXTURE)])
    assert summary["outcome_rules"]["run"] == "dry run: no outcome" and summary["outcome_rules"]["H1"] is None
    rows = [json.loads(line) for line in (tmp_path / "private" / "calls.jsonl").read_text().splitlines()]
    assert len(rows) == 12 and all(r["text"].startswith("RANKING:") for r in rows)
    public = "".join((tmp_path / name).read_text() for name in ("manifest.json", "summary.json", "analysis.md"))
    for qid in ("q1", "q2", "q3"):
        assert f'"{qid}"' not in public and f"{qid} " not in public


def test_copying_bm25_gives_zero_difference(tmp_path: Path) -> None:
    run_fixture(FixedModel(Usage(1000, 50)), tmp_path)  # plain: 1..k = BM25 order
    summary, _ = analyzer.analyze(tmp_path, FIXTURE)
    assert summary["H1_plain_minus_s0"]["mean"] == 0.0
    plain = summary["per_condition"]["plain"]
    assert plain["harm_rate"] == 0.0 and plain["copied_presentation_order_rate"] == 1.0
    assert plain["matches_bm25_order_rate"] == 1.0 and plain["stability"]["top1_agreement"] == 1.0


def test_oracle_answers_reach_the_perfect_reranker(tmp_path: Path) -> None:
    split = load_scifact(FIXTURE, "train")
    by_text = {text: doc for doc, text in split.corpus.items()}

    class OracleModel:
        def complete(self, messages, tools) -> ModelResponse:
            user = messages[-1].content
            claim = re.search(r"^Claim: (.*)$", user, re.M).group(1)
            qid = next(q for q, t in split.queries.items() if t == claim)
            shown = [by_text[t] for t in re.findall(r"^\[\d+\] (.*)$", user, re.M)]
            order = sorted(range(len(shown)), key=lambda i: -split.qrels[qid].get(shown[i], 0))
            return ModelResponse(text="RANKING: " + ", ".join(str(i + 1) for i in order), usage=Usage(1000, 50))

    run_fixture(OracleModel(), tmp_path)
    summary, _ = analyzer.analyze(tmp_path, FIXTURE)
    for condition in ("plain", "shuffled"):
        v = summary["per_condition"][condition]
        assert v["ndcg@10"] == pytest.approx(v["oracle20_ndcg@10"]) and v["harm_rate"] == 0.0


CI_POS, CI_ZERO, CI_NEG = ({"low": 0.01, "high": 0.1}, {"low": -0.01, "high": 0.1}, {"low": -0.1, "high": -0.01})


def rules(**overrides):
    base = dict(complete=True, paid=True, model_error_rate={"plain": 0.0, "shuffled": 0.0},
                format_problem_rate={"plain": 0.0, "shuffled": 0.0}, h1=CI_POS, h2=CI_ZERO, mean_cost_per_call=0.001)
    return analyzer.classify(**{**base, **overrides})


def test_outcome_rules_at_their_thresholds() -> None:
    assert rules() == {"run": "valid", "format_problem": False, "H1": "positive", "H2": "inconclusive",
                       "cost_surprise": False}
    assert rules(h1=CI_NEG, h2=CI_POS)["H1"] == "negative" and rules(h2=CI_POS)["H2"] == "shuffled better"
    assert rules(h2=CI_NEG)["H2"] == "plain better" and rules(h1=CI_ZERO)["H1"] == "inconclusive"
    assert rules(model_error_rate={"plain": 0.05, "shuffled": 0.0})["run"] == "valid"  # 5% is allowed
    invalid = rules(model_error_rate={"plain": 0.0, "shuffled": 0.051})
    assert invalid["run"].startswith("invalid") and invalid["H1"] is None
    assert rules(format_problem_rate={"plain": 0.10, "shuffled": 0.0})["format_problem"] is False
    assert rules(format_problem_rate={"plain": 0.101, "shuffled": 0.0})["format_problem"] is True
    assert rules(mean_cost_per_call=0.0036)["cost_surprise"] is False
    assert rules(mean_cost_per_call=0.00361)["cost_surprise"] is True
    assert rules(complete=False)["run"].startswith("incomplete") and rules(complete=False)["H2"] is None
