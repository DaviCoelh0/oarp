"""R1 pilot runner and analyzer with scripted models (no API key, no paid call).

A synthetic S3 pilot run (the pilot runner on the pilot tests' synthetic
dataset) is the pairing partner. Skipped without the `retrieval` and
`langgraph` extras."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

pytest.importorskip("bm25s")
pytest.importorskip("langgraph")

from oarp.core.types import ModelResponse, Usage  # noqa: E402
from oarp.envs.retrieval.data import DEFAULT_DATA_DIR, dev_partition, load_scifact  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("r1_runner", REPO / "experiments" / "retrieval_s3_r1_pilot" / "run_pilot.py")
analyzer = load("r1_analyzer", REPO / "experiments" / "retrieval_s3_r1_pilot" / "analyze.py")
helpers = load("s3_pilot_test_helpers_for_r1", REPO / "tests" / "test_retrieval_s3_pilot.py")
requires_scifact = pytest.mark.skipif(not (DEFAULT_DATA_DIR / "scifact" / "corpus.jsonl").is_file(),
                                      reason="SciFact not downloaded")


class R1Model:
    """Reformulation: "unrelated<j>" (finds the hidden relevant document of odd
    queries); rerank: numbers reversed."""

    def __init__(self, fail: Exception | None = None) -> None:
        self.fail = fail

    def complete(self, messages, tools):
        usage = Usage(12_000, 300)
        last = messages[-1].content
        if last.endswith(runner.agent_mod.REFORMULATE_INSTRUCTION):
            if self.fail is not None:
                raise self.fail
            j = re.search(r"Claim: topic(\d+)", last).group(1)
            return ModelResponse(text=f"QUERY: unrelated{j}", usage=usage, model=runner.pilot.MODEL)
        k = len(re.findall(r"^\[(\d+)\]", last, re.M))
        return ModelResponse(text="RANKING: " + ", ".join(str(i) for i in range(k, 0, -1)), usage=usage,
                             model=runner.pilot.MODEL)


def setup(tmp_path: Path) -> dict:
    """An S3 pilot run on synthetic data (2 within, 1 smoke), a check sample
    using one more spare query, and S1 records for every tune query."""
    _, data, s3_out = helpers.run(tmp_path / "s3", within_n=2, smoke_n=1)
    pilot_sample = s3_out / "private" / "sample.json"
    used = json.loads(pilot_sample.read_text())
    split = load_scifact(data, "train")
    tune = dev_partition(sorted(split.queries))["tune"]
    spare = [q for q in tune if int(q[1:]) % 2 == 0 and q not in set(used["main"]) | set(used["smoke"])]
    check_sample = tmp_path / "check_sample.json"
    assert len(spare) >= 2, "the synthetic tune half needs spare within-20 queries for both smoke phases"
    check_sample.write_text(json.dumps({"main": [], "smoke": spare[:1]}))
    pilot_s1, tau_s1 = helpers.fake_s1(tmp_path, data)
    return {"data": data, "s3_out": s3_out, "pilot_sample": pilot_sample, "check_sample": check_sample,
            "used": used, "check_smoke": spare[:1], "pilot_s1": pilot_s1, "tau_s1": tau_s1}


def run(tmp_path: Path, model=None, **kw) -> tuple[dict, dict, Path]:
    env = setup(tmp_path)
    out = tmp_path / "r1"
    manifest = runner.run_r1(model or R1Model(), data_dir=env["data"], out=out, paid=False,
                             model_settings={"model": "test"}, smoke_n=kw.pop("smoke_n", 1),
                             pilot_sample=env["pilot_sample"], check_sample=env["check_sample"],
                             log=lambda _: None, **kw)
    return manifest, env, out


def episodes(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "private" / "episodes.jsonl").read_text().splitlines()]


# --- runner ------------------------------------------------------------------------

def test_select_reuses_the_pilot_sample_and_smokes_on_unused_tune_queries() -> None:
    used = {"main": ["a", "b", "c"], "smoke": ["d"]}
    tune = {q: q in ("a", "x") for q in "abcdefgx"}
    sample = runner.select(used, check_smoke=["e"], tune_beyond=tune, smoke_n=5)
    assert sample["main"] == runner.study.hash_order(runner.ORDER_SEED, used["main"])
    assert set(sample["smoke"]) == {"f", "g"}  # not used, not checked, not beyond


def test_r1_config_is_the_s3_pilot_config_with_extra_search_once() -> None:
    from dataclasses import asdict
    s3, r1 = asdict(runner.pilot.CONFIG), asdict(runner.CONFIG)
    assert {k for k in s3 if s3[k] != r1[k]} == {"extra_search"} and r1["extra_search"] == "once"
    runner.check_pins()


def test_dry_run_pairs_every_pilot_query_and_smokes_on_a_spare_one(tmp_path: Path) -> None:
    manifest, env, out = run(tmp_path)
    assert manifest["smoke_passed"] is True and not manifest["stopped"]
    rows = episodes(out)
    main = [r["query_id"] for r in rows if r["phase"] == "main"]
    smoke = [r["query_id"] for r in rows if r["phase"] == "smoke"]
    assert sorted(main) == sorted(env["used"]["main"]) and len(smoke) == 1
    assert not set(smoke) & (set(env["used"]["main"]) | set(env["used"]["smoke"]) | set(env["check_smoke"]))
    assert all(r["config"]["extra_search"] == "once" and r["reformulation_status"] == "valid" for r in rows)
    assert not re.search(r'"q\d\d"', (out / "manifest.json").read_text())


def test_unclean_smoke_stops_and_a_decline_does_not(tmp_path: Path) -> None:
    manifest, _, _ = run(tmp_path / "a", R1Model(fail=RuntimeError("bad request")))
    assert manifest["smoke_passed"] is False and manifest["completed_episodes"]["main"] == 0
    manifest, _, _ = run(tmp_path / "b", R1Model(fail=helpers.ModelRefused("declined")))
    assert manifest["smoke_passed"] is True and manifest["completed_episodes"]["main"] > 0


def test_budget_guard_stops_between_episodes(tmp_path: Path) -> None:
    # Each scripted episode costs 2 x $0.00135 = $0.0027; cap = worst episode + $0.003 admits exactly 2.
    manifest, _, out = run(tmp_path, budget_usd=runner.EPISODE_WORST_USD + 0.003)
    assert manifest["stopped"] and "would exceed" in manifest["stop_reason"] and len(episodes(out)) == 2


def test_pins_and_strata_refusals(tmp_path: Path, monkeypatch) -> None:
    env = setup(tmp_path)
    with pytest.raises(SystemExit, match="pinned SHA-256"):
        runner.verify_pinned(paid=True, pilot_sample=env["pilot_sample"], check_sample=env["check_sample"])
    monkeypatch.setattr(runner.agent_mod, "REFORMULATE_INSTRUCTION", runner.agent_mod.REFORMULATE_INSTRUCTION + " ")
    with pytest.raises(SystemExit, match="R1 prompts"):
        runner.check_pins()
    monkeypatch.undo()
    tampered = json.loads(env["pilot_sample"].read_text())
    tampered["beyond"] = tampered["beyond"][1:]
    env["pilot_sample"].write_text(json.dumps(tampered))
    with pytest.raises(SystemExit, match="strata differ"):
        runner.run_r1(R1Model(), data_dir=env["data"], out=tmp_path / "x", paid=False, model_settings={},
                      pilot_sample=env["pilot_sample"], check_sample=env["check_sample"], log=lambda _: None)


@requires_scifact
def test_real_files_match_their_pins_and_the_design() -> None:
    runner.verify_pinned(paid=True)
    runner.check_pins()
    split = load_scifact(DEFAULT_DATA_DIR, "train")
    index = runner.BM25Index(split.corpus)
    tune = runner.pilot.strata(split, index, dev_partition(sorted(split.queries))["tune"], 20)
    used = json.loads(runner.PILOT_SAMPLE.read_text())
    check_smoke = json.loads(runner.CHECK_SAMPLE.read_text())["smoke"]
    sample = runner.select(used, check_smoke, tune)
    assert len(sample["main"]) == 163 and sum(tune[q] for q in sample["main"]) == 63
    assert len(sample["smoke"]) == 5 and all(not tune[q] for q in sample["smoke"])
    assert not set(sample["smoke"]) & (set(used["main"]) | set(used["smoke"]) | set(check_smoke))
    assert runner.study.sha256_file(analyzer.PILOT_DIR / "results" / "private" / "episodes.jsonl") == \
        analyzer.S3_EPISODES_SHA256
    assert runner.study.sha256_file(analyzer.PILOT_DIR / "analyze.py") == analyzer.PILOT_ANALYZER_SHA256


# --- analyzer ------------------------------------------------------------------------

def analyze_args(env: dict, out: Path) -> list[str]:
    return ["--results", str(out), "--pilot-results", str(env["pilot_s1"]), "--tau-results", str(env["tau_s1"]),
            "--s3-results", str(env["s3_out"]), "--data-dir", str(env["data"])]


def test_analyzer_pairs_r1_with_s3_and_finds_the_rescues(tmp_path: Path) -> None:
    manifest, env, out = run(tmp_path)
    summary = analyzer.main(analyze_args(env, out))
    assert summary["episodes"]["scored"] == manifest["completed_episodes"]["main"]
    m = summary["mechanism"]
    assert m["rescue_rate_beyond"] == {"r1": 1.0, "s3": 1.0}
    assert summary["means"]["beyond"]["r1"] == summary["means"]["beyond"]["s3"] == 1.0
    assert summary["primary_r1_minus_s3"]["mean"] == pytest.approx(0.0)
    assert m["overlap_jaccard_first_search"]["mean"] == pytest.approx(1.0)  # same query, same documents
    assert summary["outcome_rules"]["run"] == "dry run: no outcome"
    public = (out / "summary.json").read_text() + (out / "analysis.md").read_text()
    assert not re.search(r'"?q\d\d"?[ ,:]', public)


def test_analyzer_refuses_unpaired_episodes(tmp_path: Path) -> None:
    _, env, out = run(tmp_path)
    s3_path = env["s3_out"] / "private" / "episodes.jsonl"
    first = next(r["query_id"] for r in episodes(out) if r["phase"] == "main")
    s3_path.write_text("".join(line + "\n" for line in s3_path.read_text().splitlines()
                               if json.loads(line)["query_id"] != first))
    with pytest.raises(SystemExit, match="no S3 pilot episode"):
        analyzer.analyze(out, env["pilot_s1"], env["tau_s1"], env["s3_out"], env["data"])


def ci(mean: float, low: float, high: float) -> dict:
    return {"mean": mean, "low": low, "high": high}


@pytest.mark.parametrize("c, reading", [
    (ci(-0.03, -0.05, -0.01), "S3's decision adds value"),
    (ci(0.03, 0.01, 0.05), "R1 better"),
    (ci(0.0, -0.015, 0.015), "equivalent within ±0.02"),
    (ci(0.0, -0.025, 0.02), "inconclusive"),
])
def test_primary_readings(c, reading) -> None:
    assert analyzer.read_primary(c)["reading"] == reading


VALID = {"scheduled_episodes": {"main": 163}, "completed_episodes": {"main": 163}, "harness_error": None,
         "smoke_passed": True, "stopped": False}


@pytest.mark.parametrize("change, kw, run_outcome", [
    ({"harness_error": "KeyError"}, {}, "invalid (harness error)"),
    ({"smoke_passed": False, "completed_episodes": {"main": 0}}, {}, "stopped after the smoke phase (no result claimed)"),
    ({"stopped": True, "completed_episodes": {"main": 100}}, {}, "incomplete (no hypothesis claimed)"),
    ({}, {"other_error_rate": 0.06}, "invalid (non-decline model errors above 5%)"),
    ({}, {}, "valid"),
])
def test_validity_rules(change, kw, run_outcome) -> None:
    args = {"other_error_rate": 0.0, **kw}
    rules = analyzer.classify(paid=True, manifest={**VALID, **change}, decline_rate=0.0, tool_bugs=0,
                              primary=ci(0.0, -0.01, 0.01), r1_within=ci(-0.01, -0.03, 0.0), **args)
    assert rules["run"] == run_outcome
    if run_outcome == "valid":
        assert rules["primary"] == "equivalent within ±0.02" and rules["r1_within_no_harm_beyond_margin"] is False
    else:
        assert rules["primary"] is None
