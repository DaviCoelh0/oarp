"""Held-out evaluation runner and analyzer with scripted models. A SYNTHETIC
test split is built for these tests: the real SciFact test split is never
loaded here (the real-data test below touches development files only).
Skipped without the `retrieval` and `langgraph` extras."""

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


runner = load("heldout_runner", REPO / "experiments" / "retrieval_heldout" / "run_test.py")
analyzer = load("heldout_analyzer", REPO / "experiments" / "retrieval_heldout" / "analyze.py")
helpers = load("s3_pilot_test_helpers_for_heldout", REPO / "tests" / "test_retrieval_s3_pilot.py")
requires_scifact = pytest.mark.skipif(not (DEFAULT_DATA_DIR / "scifact" / "corpus.jsonl").is_file(),
                                      reason="SciFact not downloaded")
TEST_IDS = [f"x{j:02d}" for j in range(10)]


def make_data(root: Path) -> Path:
    """The pilot tests' synthetic dataset plus a synthetic test split: x<j> is
    "topic<j>", relevant c<j> (even j, within 20) or the hidden z<j> (odd j)."""
    data = helpers.make_dataset(root)
    folder = data / "scifact"
    with (folder / "queries.jsonl").open("a") as f:
        for j in range(10):
            f.write(json.dumps({"_id": f"x{j:02d}", "text": f"topic{j}"}) + "\n")
    qrels = ["query-id\tcorpus-id\tscore"] + [f"x{j:02d}\t{'c' if j % 2 == 0 else 'z'}{j:02d}\t1" for j in range(10)]
    (folder / "qrels" / "test.tsv").write_text("\n".join(qrels) + "\n")
    return data


def samples(tmp_path: Path, data: Path) -> dict[str, Path]:
    tune = dev_partition(sorted(load_scifact(data, "train").queries))["tune"]
    within = [q for q in tune if int(q[1:]) % 2 == 0]
    files = {}
    for i, name in enumerate(runner.USED_SAMPLES):
        files[name] = tmp_path / f"{name}.json"
        files[name].write_text(json.dumps({"main": within[i:i + 1], "smoke": []}))
    return files


def run(tmp_path: Path, model=None, split="test", **kw) -> tuple[dict, Path, Path, dict]:
    data = make_data(tmp_path / "data")
    used = samples(tmp_path, data)
    out = tmp_path / "out"
    manifest = runner.run_test(model or helpers.RescuingModel(), data_dir=data, out=out, paid=False,
                               model_settings={"model": "test"}, query_split=split, smoke_n=kw.pop("smoke_n", 2),
                               samples=used, log=lambda _: None, **kw)
    return manifest, data, out, used


def records(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "private" / "records.jsonl").read_text().splitlines()]


# --- runner ------------------------------------------------------------------------

def test_runs_s1_then_s3_on_every_test_query_after_a_fresh_smoke_set(tmp_path: Path, monkeypatch) -> None:
    loads = []
    real = runner.load_scifact
    monkeypatch.setattr(runner, "load_scifact", lambda d, split, **kw: loads.append((split, kw)) or real(d, split, **kw))
    manifest, data, out, used = run(tmp_path)
    assert ("test", {"include_heldout": True}) in loads
    rows = records(out)
    assert sorted(r["query_id"] for r in rows if r["phase"] == "main") == TEST_IDS
    smoke = [r["query_id"] for r in rows if r["phase"] == "smoke"]
    taken = set().union(*(set(json.loads(p.read_text())["main"]) for p in used.values()))
    assert len(smoke) == 2 and not set(smoke) & taken and all(int(q[1:]) % 2 == 0 for q in smoke)
    assert all(r["s1"]["bm25_top"] and r["s3"]["config"]["final_order"] == "forced_rerank" for r in rows)
    assert manifest["completed"] == {"smoke": 2, "main": 10} and manifest["smoke_passed"] is True
    assert not re.search(r'"[qx]\d\d"', (out / "manifest.json").read_text())


def test_the_dry_run_stand_in_never_opens_the_test_split(tmp_path: Path, monkeypatch) -> None:
    loads = []
    real = runner.load_scifact
    monkeypatch.setattr(runner, "load_scifact", lambda d, split, **kw: loads.append(split) or real(d, split, **kw))
    manifest, data, out, _ = run(tmp_path, split="check-standin")
    check = set(dev_partition(sorted(load_scifact(data, "train").queries))["check"])
    assert loads == ["train"] and {r["query_id"] for r in records(out) if r["phase"] == "main"} <= check
    with pytest.raises(SystemExit, match="paid run evaluates the test split"):
        runner.run_test(helpers.RescuingModel(), data_dir=data, out=tmp_path / "x", paid=True, model_settings={},
                        query_split="check-standin", log=lambda _: None)


def test_unclean_smoke_stops_before_the_test_split_runs(tmp_path: Path) -> None:
    manifest, _, _, _ = run(tmp_path, helpers.RescuingModel(fail_agent=RuntimeError("bad request")))
    assert manifest["smoke_passed"] is False and manifest["completed"]["main"] == 0


def test_harness_error_and_budget_stops(tmp_path: Path, monkeypatch) -> None:
    manifest, _, out, _ = run(tmp_path / "a", budget_usd=runner.QUERY_WORST_USD + 0.008)
    assert manifest["stopped"] and "would exceed" in manifest["stop_reason"]
    assert all(r["s1"] and r["s3"]["ranking"] for r in records(out))  # stopped between queries, never inside one

    def broken(*args, **kwargs):
        raise KeyError("handles")
    monkeypatch.setattr(runner.agent_mod, "run_episode", broken)
    manifest, _, _, _ = run(tmp_path / "b")
    assert manifest["harness_error"] == "KeyError: 'handles'" and manifest["completed"]["main"] == 0


# --- analyzer ------------------------------------------------------------------------

def test_analyzer_scores_s0_s1_s3_and_the_frozen_policy(tmp_path: Path) -> None:
    _, data, out, _ = run(tmp_path)
    summary, rows = analyzer.analyze(out, data, pinned=False)
    assert summary["queries"]["scored"] == len(TEST_IDS) == len(rows)  # smoke excluded
    assert summary["means"]["beyond"]["s3"] == 1.0 and summary["means"]["beyond"]["s1"] == 0.0
    for r in rows:
        assert r["route"] == analyzer.cs.route(r, analyzer.P_STAR) and r["pstar"] == r[r["route"]]
    assert summary["outcome_rules"]["run"] == "dry run: no outcome"
    assert all(k in summary for k in ("H1", "H2", "H3", "H4"))


def test_analyzer_refuses_to_open_the_real_test_split_for_unpaid_records(tmp_path: Path) -> None:
    _, data, out, _ = run(tmp_path)
    with pytest.raises(SystemExit, match="only a paid run"):
        analyzer.analyze(out, data)


def test_frozen_policy_must_match_the_search_summary(tmp_path: Path) -> None:
    summary = json.loads((analyzer.SEARCH_DIR / "results" / "summary.json").read_text())
    analyzer.check_frozen_policy(analyzer.SEARCH_DIR / "results" / "summary.json", pinned=False)
    summary["selected"]["search_tau"] = 9.0
    tampered = tmp_path / "summary.json"
    tampered.write_text(json.dumps(summary))
    with pytest.raises(SystemExit, match="P\\* differs"):
        analyzer.check_frozen_policy(tampered, pinned=False)
    with pytest.raises(SystemExit, match="pinned SHA-256"):
        analyzer.check_frozen_policy(tampered, pinned=True)


def ci(mean: float, low: float, high: float) -> dict:
    return {"mean": mean, "low": low, "high": high}


VALID = {"scheduled": {"main": 300}, "completed": {"main": 300}, "harness_error": None, "smoke_passed": True,
         "stopped": False}
GOOD = {"H1": ci(0.1, 0.08, 0.12), "H2": ci(0.03, 0.01, 0.05), "H3": ci(0.03, 0.01, 0.05),
        "H4": ci(-0.003, -0.009, 0.004)}


@pytest.mark.parametrize("change, cost_lower, claim", [
    ({}, True, True),
    ({"H4": ci(-0.005, -0.012, 0.002)}, True, False),  # H4 misses the margin
    ({}, False, False),  # not cheaper than S3
    ({"H3": ci(0.01, -0.002, 0.02)}, True, False),  # H3 fails
])
def test_optimisation_claim_needs_h3_and_h4(change, cost_lower, claim) -> None:
    rules = analyzer.classify(paid=True, manifest=VALID, other_error_rate=0.0, decline_rate=0.0,
                              h={**GOOD, **change}, cost_lower=cost_lower)
    assert rules["run"] == "valid" and rules["optimisation_claim"] is claim


@pytest.mark.parametrize("change, kw, run_outcome", [
    ({"harness_error": "KeyError"}, {}, "invalid (harness error)"),
    ({"smoke_passed": False, "completed": {"main": 0}}, {}, "stopped after the smoke phase (no result claimed)"),
    ({"stopped": True, "completed": {"main": 200}}, {}, "incomplete (no hypothesis claimed)"),
    ({}, {"other_error_rate": 0.06}, "invalid (non-decline model errors above 5%)"),
])
def test_unusable_runs_claim_nothing(change, kw, run_outcome) -> None:
    args = {"other_error_rate": 0.0, **kw}
    rules = analyzer.classify(paid=True, manifest={**VALID, **change}, decline_rate=0.0, h=GOOD, cost_lower=True,
                              **args)
    assert rules["run"] == run_outcome and rules["optimisation_claim"] is None and rules["H1"] is None


@requires_scifact
def test_real_pins_and_smoke_set_without_touching_the_test_split() -> None:
    runner.verify_pinned(paid=True)
    runner.pilot.check_pins()
    analyzer.check_frozen_policy(analyzer.SEARCH_DIR / "results" / "summary.json", pinned=True)
    dev = load_scifact(DEFAULT_DATA_DIR, "train")
    index = runner.BM25Index(dev.corpus)
    tune = runner.pilot.strata(dev, index, dev_partition(sorted(dev.queries))["tune"], 20)
    used = set()
    for path in runner.USED_SAMPLES.values():
        sample = json.loads(path.read_text())
        used |= set(sample["main"]) | set(sample["smoke"])
    smoke = runner.smoke_queries(tune, used)
    assert len(smoke) == 5 and not set(smoke) & used and all(not tune[q] for q in smoke)
