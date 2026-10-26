"""S3 check-half confirmation runner and analyzer with scripted models (no API
key, no paid call). Reuses the synthetic BEIR dataset and scripted models of
the pilot tests. Skipped without the `retrieval` and `langgraph` extras."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

pytest.importorskip("bm25s")
pytest.importorskip("langgraph")

from oarp.envs.retrieval.data import DEFAULT_DATA_DIR, dev_partition, load_scifact  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("s3_check_runner", REPO / "experiments" / "retrieval_s3_check" / "run_check.py")
analyzer = load("s3_check_analyzer", REPO / "experiments" / "retrieval_s3_check" / "analyze.py")
helpers = load("s3_pilot_test_helpers", REPO / "tests" / "test_retrieval_s3_pilot.py")
requires_scifact = pytest.mark.skipif(not (DEFAULT_DATA_DIR / "scifact" / "corpus.jsonl").is_file(),
                                      reason="SciFact not downloaded")


def setup(tmp_path: Path) -> dict:
    """Synthetic data, a pilot sample using one spare tune query, and S1 check records (S1 = BM25)."""
    data = helpers.make_dataset(tmp_path / "data")
    split = load_scifact(data, "train")
    halves = dev_partition(sorted(split.queries))
    index = runner.BM25Index(split.corpus)
    tune_within = [q for q in halves["tune"] if int(q[1:]) % 2 == 0]
    pilot_sample = tmp_path / "pilot_sample.json"
    pilot_sample.write_text(json.dumps({"main": tune_within[:1], "smoke": tune_within[1:2]}))
    s1 = tmp_path / "s1_check"
    (s1 / "private").mkdir(parents=True)
    (s1 / "manifest.json").write_text("{}")
    (s1 / "private" / "sample.json").write_text("{}")
    rows = []
    for q in halves["check"]:
        top = [d for d, _ in index.rank(split.queries[q], 20)]
        rows.append({"pass": 1, "condition": "plain", "query_id": q, "bm25_top": top, "ranking_top": top,
                     "estimated_usd": 0.0013, "latency_s": 4.0})
    rows.append({**rows[0], "pass": 2})  # later passes are ignored
    (s1 / "private" / "calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {"data": data, "halves": halves, "pilot_sample": pilot_sample, "s1": s1,
            "pilot_used": set(tune_within[:2])}


def run(tmp_path: Path, model=None, **kw) -> tuple[dict, dict, Path]:
    env = setup(tmp_path)
    out = tmp_path / "out"
    manifest = runner.run_check(model or helpers.RescuingModel(), data_dir=env["data"], out=out, paid=False,
                                model_settings={"model": "test"}, smoke_n=kw.pop("smoke_n", 2),
                                pilot_sample=env["pilot_sample"], s1_check=env["s1"], log=lambda _: None, **kw)
    return manifest, env, out


def episodes(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "private" / "episodes.jsonl").read_text().splitlines()]


# --- runner ------------------------------------------------------------------------

def test_select_runs_every_check_query_and_smokes_on_spare_tune_queries() -> None:
    check = {f"c{i}": i % 4 == 0 for i in range(12)}
    tune = {f"t{i}": i % 3 == 0 for i in range(12)}
    sample = runner.select(check, tune, pilot_used={"t1", "t2"}, smoke_n=3)
    assert sample["main"] == runner.study.hash_order(runner.ORDER_SEED, list(check))
    spare = [q for q, b in tune.items() if not b and q not in {"t1", "t2"}]
    assert sample["smoke"] == runner.study.hash_order(runner.ORDER_SEED, spare)[:3]


def test_dry_run_runs_smoke_on_tune_then_every_check_query(tmp_path: Path) -> None:
    manifest, env, out = run(tmp_path)
    check, tune = set(env["halves"]["check"]), set(env["halves"]["tune"])
    assert manifest["smoke_passed"] is True and not manifest["stopped"]
    assert manifest["completed_episodes"] == {"smoke": 2, "main": len(check)}
    rows = episodes(out)
    smoke = [r["query_id"] for r in rows if r["phase"] == "smoke"]
    assert set(smoke) <= tune and not set(smoke) & env["pilot_used"]
    assert all(int(q[1:]) % 2 == 0 for q in smoke)  # within-20 tune queries only
    assert {r["query_id"] for r in rows if r["phase"] == "main"} == check
    assert all(r["config"]["final_order"] == "forced_rerank" for r in rows)
    assert not re.search(r'"q\d\d"', (out / "manifest.json").read_text())


def test_unclean_smoke_episode_stops_before_the_check_half(tmp_path: Path) -> None:
    manifest, _, _ = run(tmp_path, helpers.RescuingModel(fail_agent=RuntimeError("bad request")))
    assert manifest["smoke_passed"] is False and manifest["completed_episodes"] == {"smoke": 1, "main": 0}


def test_harness_error_stops_the_run(tmp_path: Path, monkeypatch) -> None:
    def broken(*args, **kwargs):
        raise KeyError("handles")
    monkeypatch.setattr(runner.agent_mod, "run_episode", broken)
    manifest, _, _ = run(tmp_path)
    assert manifest["harness_error"] == "KeyError: 'handles'" and manifest["completed_episodes"]["main"] == 0


def test_budget_guard_stops_between_episodes(tmp_path: Path) -> None:
    manifest, _, out = run(tmp_path, budget_usd=runner.pilot.EPISODE_WORST_USD + 0.01)
    assert manifest["stopped"] and "would exceed" in manifest["stop_reason"]
    assert len(episodes(out)) == 3 and all(r["ranking"] for r in episodes(out))


def test_paid_mode_refuses_reused_files_that_differ_from_their_pins(tmp_path: Path) -> None:
    env = setup(tmp_path)
    runner.verify_pinned(paid=False, pilot_sample=env["pilot_sample"], s1_check=env["s1"])  # dry run: presence only
    with pytest.raises(SystemExit, match="pinned SHA-256"):
        runner.verify_pinned(paid=True, pilot_sample=env["pilot_sample"], s1_check=env["s1"])
    with pytest.raises(SystemExit, match="missing"):
        runner.verify_pinned(paid=False, pilot_sample=tmp_path / "nope.json", s1_check=env["s1"])


@requires_scifact
def test_real_files_match_their_pins_and_the_contract() -> None:
    runner.verify_pinned(paid=True)  # pilot runner, pilot sample, S1 check records
    runner.pilot.check_pins()
    split = load_scifact(DEFAULT_DATA_DIR, "train")
    index = runner.BM25Index(split.corpus)
    halves = dev_partition(sorted(split.queries))
    check = runner.pilot.strata(split, index, halves["check"], 20)
    tune = runner.pilot.strata(split, index, halves["tune"], 20)
    assert (len(check), sum(check.values())) == (404, 55)
    used = json.loads(runner.PILOT_SAMPLE.read_text())
    sample = runner.select(check, tune, set(used["main"]) | set(used["smoke"]))
    assert len(sample["main"]) == 404 and len(sample["smoke"]) == 5
    assert not set(sample["smoke"]) & (set(used["main"]) | set(used["smoke"]))
    assert all(not tune[q] for q in sample["smoke"])


# --- analyzer ------------------------------------------------------------------------

def test_analyzer_scores_check_episodes_only_and_finds_the_rescues(tmp_path: Path) -> None:
    manifest, env, out = run(tmp_path)
    summary = analyzer.main(["--results", str(out), "--s1-check", str(env["s1"]), "--data-dir", str(env["data"]),
                             "--pilot-summary", str(tmp_path / "absent.json")])
    assert summary["episodes"]["scored"] == len(env["halves"]["check"])  # smoke excluded
    assert summary["mechanism"]["rescue_rate_beyond"] == 1.0
    assert summary["means"]["beyond"]["s1"] == 0.0 and summary["C3_beyond_s3fr_minus_s1"]["mean"] == 1.0
    assert summary["outcome_rules"] == {"run": "dry run: no outcome", "outcome": None, "decline_warning": False}
    public = (out / "summary.json").read_text() + (out / "analysis.md").read_text()
    assert not re.search(r'"?q\d\d"?[ ,:]', public)


def test_analyzer_refuses_episodes_without_an_s1_check_record(tmp_path: Path) -> None:
    _, env, out = run(tmp_path)
    path = env["s1"] / "private" / "calls.jsonl"
    first = next(r["query_id"] for r in episodes(out) if r["phase"] == "main")
    path.write_text("".join(line + "\n" for line in path.read_text().splitlines()
                            if json.loads(line)["query_id"] != first))
    with pytest.raises(SystemExit, match="no S1 check record"):
        analyzer.analyze(out, env["s1"], env["data"])


def ci(mean: float, low: float, high: float) -> dict:
    return {"mean": mean, "low": low, "high": high}


VALID = {"scheduled_episodes": {"main": 404}, "completed_episodes": {"main": 404}, "harness_error": None,
         "smoke_passed": True, "stopped": False}


@pytest.mark.parametrize("c1, c2, outcome", [
    (ci(0.04, 0.02, 0.06), ci(0.01, -0.005, 0.03), "confirmed"),
    (ci(0.04, 0.02, 0.06), ci(-0.02, -0.035, -0.005), "confirmed with harm"),
    (ci(0.01, -0.002, 0.02), ci(0.01, -0.005, 0.03), "not confirmed"),
])
def test_outcome_rule(c1, c2, outcome) -> None:
    rules = analyzer.classify(paid=True, manifest=VALID, other_error_rate=0.0, decline_rate=0.0, tool_bugs=0,
                              c1=c1, c2=c2)
    assert rules["run"] == "valid" and rules["outcome"] == outcome


@pytest.mark.parametrize("change, kw, run_outcome", [
    ({"harness_error": "KeyError"}, {}, "invalid (harness error)"),
    ({}, {"tool_bugs": 1}, "invalid (harness error)"),
    ({"smoke_passed": False, "completed_episodes": {"main": 0}}, {}, "stopped after the smoke phase (no result claimed)"),
    ({"stopped": True, "completed_episodes": {"main": 300}}, {}, "incomplete (no hypothesis claimed)"),
    ({}, {"other_error_rate": 0.06}, "invalid (non-decline model errors above 5%)"),
])
def test_unusable_runs_claim_no_outcome(change, kw, run_outcome) -> None:
    args = {"other_error_rate": 0.0, "tool_bugs": 0, **kw}
    rules = analyzer.classify(paid=True, manifest={**VALID, **change}, decline_rate=0.0,
                              c1=ci(0.1, 0.05, 0.2), c2=ci(0.0, -0.01, 0.01), **args)
    assert rules["run"] == run_outcome and rules["outcome"] is None


def test_decline_warning() -> None:
    rules = analyzer.classify(paid=True, manifest=VALID, other_error_rate=0.0, decline_rate=0.11, tool_bugs=0,
                              c1=ci(0.1, 0.05, 0.2), c2=ci(0.0, -0.01, 0.01))
    assert rules["decline_warning"] and rules["outcome"] == "confirmed"
