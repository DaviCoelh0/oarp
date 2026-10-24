"""S2 threshold study: selection rule, runner and analyzer, free: scripted
models, a fake Anthropic client, and dry-run S1 pilot and check runs on the
small BEIR-format fixture. No network, no paid call."""

import importlib.util
import json
import math
import re
import shutil
from pathlib import Path

import pytest

pytest.importorskip("bm25s")

from oarp.core.pricing import estimate_cost  # noqa: E402
from oarp.core.types import ModelResponse, Usage  # noqa: E402
from oarp.envs.retrieval.analysis import fewest_reranks_within  # noqa: E402
from oarp.envs.retrieval.data import dev_partition, load_scifact  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "retrieval"


def _load(folder: str, name: str):
    spec = importlib.util.spec_from_file_location(f"tau_{folder}_{name}", REPO / "experiments" / folder / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot_runner = _load("retrieval_s1_pilot", "run_pilot")
check_runner = _load("retrieval_s1_check", "run_check")
runner = _load("retrieval_s2_tau", "run_tune")
analyzer = _load("retrieval_s2_tau", "analyze")
HALVES = dev_partition(sorted(load_scifact(FIXTURE, "train").queries))
QUIET = {"log": lambda _: None}


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, Path]:
    """Dry-run S1 pilot (sample of 1 tune query) and dry-run S1 check run on the fixture."""
    pilot, check = tmp_path / "pilot", tmp_path / "check"
    pilot_runner.run_pilot(pilot_runner.DryRunModel(), data_dir=FIXTURE, out=pilot, paid=False,
                           model_settings={}, sample_size=1, stability=1, **QUIET)
    check_runner.run_check(check_runner.DryRunModel(), data_dir=FIXTURE, out=check, paid=False,
                           model_settings={}, **QUIET)
    return {"pilot": pilot, "check": check, "out": tmp_path / "out"}


class FixedModel:
    def __init__(self, usage=Usage(1000, 50)) -> None:
        self.usage, self.calls = usage, 0

    def complete(self, messages, tools) -> ModelResponse:
        self.calls += 1
        k = len(re.findall(r"^\[(\d+)\]", messages[-1].content, re.M))
        return ModelResponse(text="RANKING: " + ", ".join(map(str, range(1, k + 1))), usage=self.usage, model="fake")


def run(inputs: dict, model=None, **kw) -> dict:
    return runner.run_tune(model or runner.DryRunModel(), data_dir=FIXTURE, out=inputs["out"], paid=False,
                           model_settings={}, pilot_results=inputs["pilot"], **QUIET, **kw)


def analyze(inputs: dict) -> dict:
    return analyzer.analyze(inputs["out"], inputs["pilot"], inputs["check"], FIXTURE)[0]


# --- the selection rule ---------------------------------------------------------------------

def test_selection_rule_by_hand() -> None:
    gaps, s0, s1 = {"a": 0.1, "b": 0.5, "c": 0.9}, {"a": 0, "b": 1, "c": 1}, {"a": 1, "b": 1, "c": 0.9}
    best = fewest_reranks_within(gaps, s0, s1, tolerance=0)
    assert (best["tau"], best["rate"]) == (0.5, pytest.approx(1 / 3))  # reranking a only beats S1
    assert len(best["curve"]) == 5  # -inf, three gaps, +inf


def test_only_always_rerank_qualifies() -> None:
    gaps, s0, s1 = {"a": 0.1, "b": 0.5}, {"a": 0, "b": 0}, {"a": 1, "b": 1}
    best = fewest_reranks_within(gaps, s0, s1, tolerance=0.01)
    assert best["tau"] == math.inf and best["rate"] == 1.0


def test_ties_go_to_the_smaller_tau_and_tolerance_allows_a_loss() -> None:
    # S1 mean 1.0. tau = -inf and tau = 0.2 both rerank nothing (gap < tau): mean 0.933, loss 0.067.
    # tau = 0.7 reranks a and b only: same loss, at rate 2/3. tau = +inf is S1.
    gaps, s0, s1 = {"a": 0.2, "b": 0.2, "c": 0.7}, {"a": 1, "b": 1, "c": 0.8}, {"a": 1, "b": 1, "c": 1}
    loose = fewest_reranks_within(gaps, s0, s1, 0.1)
    assert loose["tau"] == -math.inf and loose["rate"] == 0.0  # tie at rate 0: the smaller tau
    assert loose["minus_s1"] == pytest.approx(-0.2 / 3)
    assert fewest_reranks_within(gaps, s0, s1, 0.05)["tau"] == math.inf  # the loss exceeds the tolerance
    assert fewest_reranks_within(gaps, s0, s1, 0.0)["tau"] == math.inf  # tolerance 0: no loss allowed


# --- runner ----------------------------------------------------------------------------------

def test_queries_are_tune_minus_pilot_in_hash_order() -> None:
    tune, pilot = [f"t{i}" for i in range(10)], ["t1", "t5"]
    chosen = runner.select_queries(tune, pilot)
    assert sorted(chosen) == sorted(set(tune) - set(pilot))
    assert chosen == runner.study.hash_order(runner.ORDER_SEED, chosen)
    with pytest.raises(SystemExit, match="not inside the tune half"):
        runner.select_queries(tune, ["x9"])


def test_runner_calls_only_tune_queries_outside_the_pilot(inputs: dict, monkeypatch) -> None:
    loaded = []

    def spy(d, split, **kw):
        loaded.append((split, kw))
        return load_scifact(d, split, **kw)

    monkeypatch.setattr(runner, "load_scifact", spy)
    run(inputs)
    assert loaded == [("train", {})]
    called = json.loads((inputs["out"] / "private" / "sample.json").read_text())["queries"]
    pilot = json.loads((inputs["pilot"] / "private" / "sample.json").read_text())["sample"]
    assert called and set(called) == set(HALVES["tune"]) - set(pilot) and not set(called) & set(HALVES["check"])


def test_refuses_a_pilot_with_other_prompts_or_config(inputs: dict) -> None:
    path = inputs["pilot"] / "manifest.json"
    good = json.loads(path.read_text())
    path.write_text(json.dumps({**good, "prompt_sha256": {"rerank_system": "x", "rerank_instruction": "y"}}))
    with pytest.raises(SystemExit, match="prompts differ"):
        runner.load_pilot(inputs["pilot"], paid=False)
    bad = json.loads(json.dumps(good))
    bad["conditions"]["plain"]["rerank_config"]["k"] = 10
    path.write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="RerankConfig differs"):
        runner.load_pilot(inputs["pilot"], paid=False)


def test_paid_mode_refuses_unpinned_pilot_files(inputs: dict) -> None:
    with pytest.raises(SystemExit, match="pinned SHA-256"):
        runner.load_pilot(inputs["pilot"], paid=True)


def test_budget_guard_in_the_runner(inputs: dict, tmp_path: Path) -> None:
    manifest = run(inputs, FixedModel(), budget_usd=0.003)  # below one worst-case call
    assert manifest["completed_calls"] == 0 and manifest["stopped_by_budget"]
    inputs = {**inputs, "out": tmp_path / "out2"}
    manifest = run(inputs, FixedModel(Usage(50_000, 4_000)))  # $0.007 > the $0.004 bound
    assert manifest["calls_above_worst_case"] == 1 and manifest["stopped_by_budget"]


# --- analyzer --------------------------------------------------------------------------------

def test_dry_run_end_to_end_is_labelled_not_blind_and_keeps_ids_private(inputs: dict) -> None:
    manifest = runner.main(["--dry-run", "--data-dir", str(FIXTURE), "--results", str(inputs["out"]),
                            "--pilot-results", str(inputs["pilot"])])
    assert manifest["mode"].startswith("dry-run") and manifest["completed_calls"] == manifest["scheduled_calls"]
    summary = analyzer.main(["--results", str(inputs["out"]), "--pilot-results", str(inputs["pilot"]),
                             "--check-results", str(inputs["check"]), "--data-dir", str(FIXTURE)])
    assert summary["outcome_rules"]["run"] == "dry run: no outcome" and summary["outcome_rules"]["H_S2"] is None
    assert summary["tune"]["queries"] == len(HALVES["tune"]) and summary["check_confirmation"]["queries"] == len(HALVES["check"])
    out = inputs["out"]
    public = "".join((out / n).read_text() for n in ("manifest.json", "summary.json", "analysis.md"))
    assert "NOT BLIND" in public
    for qid in ("q1", "q2", "q3"):
        assert f'"{qid}"' not in public and f"{qid} " not in public


def test_threshold_extremes_give_s0_and_s1_with_matching_costs(inputs: dict, monkeypatch) -> None:
    run(inputs)
    rerank_cost = estimate_cost("claude-haiku-5-5", Usage(1000, 50))  # the dry-run check records' usage
    for tau, expect in ((-math.inf, "s0"), (math.inf, "s1")):
        monkeypatch.setattr(analyzer, "fewest_reranks_within",
                            lambda *a, tau=tau, **k: {"tau": tau, "rate": float(tau > 0), "minus_s1": 0.0, "curve": []})
        c = analyze(inputs)["check_confirmation"]
        assert c["s2"] == pytest.approx(c[expect])
        assert c["cost_per_query_usd"] == pytest.approx(0.0 if expect == "s0" else rerank_cost)
        assert c["s1_cost_per_query_usd"] == pytest.approx(rerank_cost)


def test_analyzer_refuses_changed_or_incomplete_inputs(inputs: dict, tmp_path: Path) -> None:
    run(inputs)
    manifest_path = inputs["out"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_path.write_text(json.dumps({**manifest, "mode": "paid"}))
    with pytest.raises(SystemExit, match="pinned SHA-256"):
        analyze(inputs)  # fixture pilot and check files are not the pinned real ones
    manifest_path.write_text(json.dumps(manifest))
    broken = tmp_path / "broken_check"
    shutil.copytree(inputs["check"], broken)
    (broken / "private" / "calls.jsonl").write_text("")
    with pytest.raises(SystemExit, match="cover exactly the check half"):
        analyzer.analyze(inputs["out"], inputs["pilot"], broken, FIXTURE)


def test_the_real_reused_files_match_their_pins() -> None:
    if not (analyzer.CHECK_RESULTS / "private" / "calls.jsonl").is_file():
        pytest.skip("private study records not present on this machine")
    assert analyzer.verify_reused("pilot", analyzer.PILOT_RESULTS, paid=True)
    assert analyzer.verify_reused("check", analyzer.CHECK_RESULTS, paid=True)
    assert runner.load_pilot(runner.PILOT_RESULTS, paid=True)["commit"]


CI_OK, CI_LOW = {"low": -0.019, "high": 0.01}, {"low": -0.021, "high": 0.01}


def rules(**overrides):
    base = dict(complete=True, paid=True, other_error_rate=0.0, decline_rate=0.0, tau_star=0.2, h_s2=CI_OK)
    return analyzer.classify(**{**base, **overrides})


def test_outcome_rules_at_their_thresholds() -> None:
    assert rules()["H_S2"] == "non-inferior (not blind)" and rules()["run"] == "valid"
    assert rules(h_s2=CI_LOW)["H_S2"] == "not shown"
    assert rules(tau_star=math.inf)["H_S2"] == "no saving possible" and rules(tau_star=math.inf)["no_saving_possible"]
    assert rules(other_error_rate=0.05)["run"] == "valid" and rules(other_error_rate=0.051)["H_S2"] is None
    assert rules(decline_rate=0.10)["decline_warning"] is False and rules(decline_rate=0.101)["decline_warning"]
    assert rules(complete=False)["run"].startswith("incomplete") and rules(paid=False)["H_S2"] is None
    assert "NOT BLIND" in rules()["blindness"]


# --- the real adapter path, with a fake client -------------------------------------------------

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


def test_paid_path_request_shape_and_no_key_leak(inputs: dict, tmp_path: Path, monkeypatch, capsys) -> None:
    secret = "sk-ant-dummy-NEVER-WRITE-ME"
    (tmp_path / ".env").write_text(f"ANTHROPIC_API_KEY={secret}\n")
    clients = []
    real = anthropic_model.AnthropicModel

    def fake_model(settings, *, api_key=None, client=None):
        assert api_key == secret
        clients.append(FakeClient())
        return real(settings, client=clients[-1])

    real_load = runner.load_pilot
    monkeypatch.setattr(anthropic_model, "AnthropicModel", fake_model)
    monkeypatch.setattr(runner.study, "preflight", lambda *a, **kw: {"git": None})
    monkeypatch.setattr(runner, "load_pilot", lambda p, paid: real_load(p, paid=False))  # fixture pilot is unpinned
    monkeypatch.setattr(runner, "EXPECTED_QUERIES", len(set(HALVES["tune"])) - 1)
    out = inputs["out"]
    manifest = runner.main(["--env-file", str(tmp_path / ".env"), "--data-dir", str(FIXTURE), "--results", str(out),
                            "--pilot-results", str(inputs["pilot"])])
    assert manifest["mode"] == "paid" and manifest["completed_calls"] == manifest["scheduled_calls"] >= 1
    request = clients[0].requests[0]
    assert "tools" not in request and request["output_config"] == {"effort": "medium"}
    assert request["max_tokens"] == 4096 and not {"temperature", "top_p", "top_k"} & set(request)
    written = "".join(p.read_text() for p in out.rglob("*") if p.is_file())
    assert secret not in written and secret not in capsys.readouterr().out
