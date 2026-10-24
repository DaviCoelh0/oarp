"""S2b gate runner and analyzer, free: scripted models, a fake Anthropic client,
and a dry-run S1 pilot on the small BEIR-format fixture. No network, no paid call."""

import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest

pytest.importorskip("bm25s")

from oarp.core.types import ModelResponse, Usage  # noqa: E402
from oarp.core.pricing import estimate_cost  # noqa: E402
from oarp.envs.retrieval.data import load_scifact  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "retrieval"


def _load(folder: str, name: str):
    spec = importlib.util.spec_from_file_location(f"{folder}_{name}", REPO / "experiments" / folder / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot_runner = _load("retrieval_s1_pilot", "run_pilot")
runner = _load("retrieval_s2b_pilot", "run_gate")
analyzer = _load("retrieval_s2b_pilot", "analyze")
SPLIT = load_scifact(FIXTURE, "train")
BY_TEXT = {text: doc for doc, text in SPLIT.corpus.items()}


@pytest.fixture
def pilot(tmp_path: Path) -> Path:
    """A dry-run S1 pilot on the fixture: the gate study's read-only input."""
    out = tmp_path / "pilot"
    pilot_runner.run_pilot(pilot_runner.DryRunModel(), data_dir=FIXTURE, out=out, paid=False,
                           model_settings={"model": "dry-run"}, log=lambda _: None)
    return out


class GateModel:
    """Answers with `decide(query_id, shown doc ids)`; records what it was shown."""

    def __init__(self, decide, usage=Usage(1700, 20)) -> None:
        self.decide, self.usage, self.shown = decide, usage, []

    def complete(self, messages, tools) -> ModelResponse:
        user = messages[-1].content
        qid = next(q for q, t in SPLIT.queries.items() if t == re.search(r"^Claim: (.*)$", user, re.M).group(1))
        shown = [BY_TEXT[t] for t in re.findall(r"^\[\d+\] (.*)$", user, re.M)]
        self.shown.append((qid, shown, user))
        answer = "yes" if self.decide(qid, shown) else "no"
        return ModelResponse(text=f"ANSWER: {answer}", usage=self.usage, model="fake", stop_reason="end_turn")


def run_fixture(models, pilot: Path, out: Path, **kw) -> dict:
    return runner.run_gate(models, data_dir=FIXTURE, out=out, paid=False, model_settings={}, pilot_results=pilot,
                           log=lambda _: None, **kw)


def both(model) -> dict:
    return {"medium": model, "low": model}


# --- inputs from the S1 pilot ---------------------------------------------------------------

def test_refuses_missing_changed_or_unpaid_pilot_files(pilot: Path, tmp_path: Path) -> None:
    runner.load_pilot(pilot, paid=False)  # a dry-run pilot is fine for a dry run
    with pytest.raises(SystemExit, match="not a complete paid run"):
        runner.load_pilot(pilot, paid=True)
    manifest = json.loads((pilot / "manifest.json").read_text())
    (pilot / "manifest.json").write_text(json.dumps({**manifest, "mode": "paid"}))
    with pytest.raises(SystemExit, match="differ from the pinned"):
        runner.load_pilot(pilot, paid=True)
    broken = tmp_path / "broken"
    shutil.copytree(pilot, broken)
    (broken / "private" / "calls.jsonl").unlink()
    with pytest.raises(SystemExit, match="missing"):
        runner.load_pilot(broken, paid=False)


def test_the_real_pilot_files_match_their_pinned_hashes() -> None:
    if not (runner.PILOT_RESULTS / "private" / "calls.jsonl").is_file():
        pytest.skip("S1 pilot private records not present on this machine")
    loaded = runner.load_pilot(runner.PILOT_RESULTS, paid=True)
    assert len(loaded["sample"]) == 100


def test_schedule_shape_and_effort_order() -> None:
    sample = [f"t{i}" for i in range(100)]
    schedule = runner.build_schedule(sample)
    assert len(schedule) == 600 and [e["index"] for e in schedule] == list(range(1, 601))
    for i in range(0, 600, 2):
        a, b = schedule[i], schedule[i + 1]
        assert a["query_id"] == b["query_id"] and a["pass"] == b["pass"] and {a["effort"], b["effort"]} == {"medium", "low"}
        assert (a["effort"], b["effort"]) == runner.study.first_of_pair(runner.ORDER_SEED, a["query_id"], ("medium", "low"))
    assert {e["pass"] for e in schedule} == {1, 2, 3}


def test_gate_sees_exactly_the_recorded_top_three(pilot: Path, tmp_path: Path) -> None:
    model = GateModel(lambda q, shown: True)
    run_fixture(both(model), pilot, tmp_path / "out")
    rows = [json.loads(line) for line in (pilot / "private" / "calls.jsonl").read_text().splitlines()]
    top = {r["query_id"]: r["bm25_top"][:3] for r in rows if r["pass"] == 1 and r["condition"] == "plain"}
    assert model.shown and all(shown == top[q] for q, shown, _ in model.shown)
    assert not any(doc in user for _, _, user in model.shown for doc in SPLIT.corpus)


def test_refuses_a_recorded_top_three_that_differs_from_bm25(pilot: Path, tmp_path: Path) -> None:
    path = pilot / "private" / "calls.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for r in rows:
        r["bm25_top"] = list(reversed(r["bm25_top"]))
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SystemExit, match="fresh BM25"):
        run_fixture(both(GateModel(lambda q, s: True)), pilot, tmp_path / "out")


# --- analyzer -------------------------------------------------------------------------------

def test_always_yes_is_s0_and_always_no_is_s1(pilot: Path, tmp_path: Path) -> None:
    run_fixture(both(GateModel(lambda q, s: True)), pilot, tmp_path / "yes")
    yes, _ = analyzer.analyze(tmp_path / "yes", pilot, FIXTURE)
    st = yes["strategies"]
    assert st["S2b_medium"]["rerank_rate"] == 0.0 and st["S2b_medium"]["ndcg@10"] == st["S0"]["ndcg@10"]
    run_fixture(both(GateModel(lambda q, s: False)), pilot, tmp_path / "no")
    no, _ = analyzer.analyze(tmp_path / "no", pilot, FIXTURE)
    st = no["strategies"]
    assert st["S2b_medium"]["rerank_rate"] == 1.0 and st["S2b_medium"]["ndcg@10"] == st["S1"]["ndcg@10"]


def test_cost_per_query_is_gate_plus_reranks(pilot: Path, tmp_path: Path) -> None:
    run_fixture(both(GateModel(lambda q, s: False, Usage(1700, 20))), pilot, tmp_path / "out")
    summary, _ = analyzer.analyze(tmp_path / "out", pilot, FIXTURE)
    gate_cost = estimate_cost("claude-haiku-5-5", Usage(1700, 20))
    rerank_cost = estimate_cost("claude-haiku-5-5", Usage(1000, 50))  # the dry-run pilot's scripted usage
    assert summary["strategies"]["S2b_medium"]["cost_per_query_usd"] == pytest.approx(gate_cost + rerank_cost)
    assert summary["strategies"]["S1"]["cost_per_query_usd"] == pytest.approx(rerank_cost)
    assert summary["strategies"]["S0"]["cost_per_query_usd"] == 0.0


def test_a_perfect_gate_is_right_every_time(pilot: Path, tmp_path: Path) -> None:
    perfect = GateModel(lambda q, shown: SPLIT.qrels[q].get(shown[0], 0) > 0)
    run_fixture(both(perfect), pilot, tmp_path / "out")
    summary, _ = analyzer.analyze(tmp_path / "out", pilot, FIXTURE)
    g = summary["gates"]["medium"]
    assert g["correct_rate"] == 1.0 and g["flip_rate"] == 0.0
    assert summary["strategies"]["S2b_medium"]["ndcg@10"] == summary["strategies"]["oracle_gate"]["ndcg@10"]
    assert g["precision"] in (1.0, None) and g["recall"] in (1.0, None)


def test_flip_rate_counts_changing_decisions(pilot: Path, tmp_path: Path) -> None:
    seen: dict[str, int] = {}

    def alternate(q, shown):  # yes, no, yes for each query across its passes
        seen[q] = seen.get(q, 0) + 1
        return seen[q] % 2 == 1

    run_fixture({"medium": GateModel(alternate), "low": GateModel(lambda q, s: True)}, pilot, tmp_path / "out")
    summary, _ = analyzer.analyze(tmp_path / "out", pilot, FIXTURE)
    assert summary["gates"]["low"]["flip_rate"] == 0.0 and summary["gates"]["medium"]["flip_rate"] == 1.0


def test_dry_run_end_to_end_keeps_ids_private(pilot: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    manifest = runner.main(["--dry-run", "--data-dir", str(FIXTURE), "--results", str(out), "--pilot-results", str(pilot)])
    assert manifest["mode"].startswith("dry-run") and manifest["completed_calls"] == manifest["scheduled_calls"]
    summary = analyzer.main(["--results", str(out), "--pilot-results", str(pilot), "--data-dir", str(FIXTURE)])
    assert summary["outcome_rules"]["run"] == "dry run: no outcome" and summary["outcome_rules"]["primary"] is None
    public = "".join((out / n).read_text() for n in ("manifest.json", "summary.json", "analysis.md"))
    for qid in ("q1", "q2", "q3"):
        assert f'"{qid}"' not in public and f"{qid} " not in public


CI_POS, CI_ZERO, CI_NEG = ({"low": 0.01, "high": 0.1}, {"low": -0.01, "high": 0.1}, {"low": -0.1, "high": -0.01})


def rules(**overrides):
    base = dict(complete=True, paid=True, other_error_rate={"medium": 0.0, "low": 0.0},
                unparseable_rate={"medium": 0.0, "low": 0.0}, primary=CI_POS, gate_cost_per_query=0.0008,
                s1_cost_per_query=0.0013, flip_medium=0.1)
    return analyzer.classify(**{**base, **overrides})


def test_outcome_rules_at_their_thresholds() -> None:
    assert rules() == {"run": "valid", "format_problem": False, "primary": "gate better", "no_saving": False,
                       "unstable": False}
    assert rules(primary=CI_ZERO)["primary"] == "gate not better" and rules(primary=CI_NEG)["primary"] == "gate worse"
    assert rules(other_error_rate={"medium": 0.05, "low": 0.0})["run"] == "valid"
    assert rules(other_error_rate={"medium": 0.0, "low": 0.051})["run"].startswith("invalid")
    assert rules(unparseable_rate={"medium": 0.10, "low": 0.0})["format_problem"] is False
    assert rules(unparseable_rate={"medium": 0.101, "low": 0.0})["format_problem"] is True
    assert rules(gate_cost_per_query=0.0013)["no_saving"] is True
    assert rules(flip_medium=0.20)["unstable"] is False and rules(flip_medium=0.201)["unstable"] is True
    assert rules(complete=False)["primary"] is None and rules(paid=False)["run"] == "dry run: no outcome"


# --- the real adapter path, with a fake client ----------------------------------------------

anthropic = pytest.importorskip("anthropic")
from anthropic.types import Message as ApiMessage  # noqa: E402

from oarp.providers import anthropic_model  # noqa: E402


class FakeClient:
    def __init__(self) -> None:
        self.requests = []
        self.messages = self

    def create(self, **request):
        self.requests.append(request)
        return ApiMessage.model_validate({
            "id": "m", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
            "content": [{"type": "text", "text": "ANSWER: no"}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1700, "output_tokens": 200}})


def test_each_effort_reaches_the_request_and_the_key_never_leaks(pilot: Path, tmp_path: Path, monkeypatch,
                                                                 capsys) -> None:
    secret = "sk-ant-dummy-NEVER-WRITE-ME"
    (tmp_path / ".env").write_text(f"ANTHROPIC_API_KEY={secret}\n")
    clients = {}
    real = anthropic_model.AnthropicModel

    def fake_model(settings, *, api_key=None, client=None):
        assert api_key == secret
        clients[settings.effort] = FakeClient()
        return real(settings, client=clients[settings.effort])

    monkeypatch.setattr(anthropic_model, "AnthropicModel", fake_model)
    monkeypatch.setattr(runner.study, "preflight", lambda *a, **kw: {"git": None})
    real_load = runner.load_pilot
    monkeypatch.setattr(runner, "load_pilot", lambda p, paid: real_load(p, paid=False))  # the fixture pilot is a dry run
    out = tmp_path / "out"
    manifest = runner.main(["--env-file", str(tmp_path / ".env"), "--data-dir", str(FIXTURE), "--results", str(out),
                            "--pilot-results", str(pilot)])
    assert manifest["mode"] == "paid" and set(clients) == {"medium", "low"}
    medium, low = clients["medium"].requests[0], clients["low"].requests[0]
    assert medium["output_config"] == {"effort": "medium"} and low["output_config"] == {"effort": "low"}
    # Both first requests are for the same query: identical apart from the effort.
    assert {k: v for k, v in medium.items() if k != "output_config"} == {k: v for k, v in low.items() if k != "output_config"}
    assert "tools" not in medium and not {"temperature", "top_p", "top_k"} & set(medium)
    written = "".join(p.read_text() for p in out.rglob("*") if p.is_file())
    assert secret not in written and secret not in capsys.readouterr().out

