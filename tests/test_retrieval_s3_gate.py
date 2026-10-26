"""Offline gate study: gating, reweighting, the selection rule, the check
outcome rule, pins and privacy, on synthetic records; and, with the downloaded
data, the real pinned inputs. Skipped without the `retrieval` extra."""

import importlib.util
import json
import math
import re
from pathlib import Path

import pytest

pytest.importorskip("bm25s")

from oarp.envs.retrieval.data import DEFAULT_DATA_DIR, dev_partition, load_scifact  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = load("s3_gate", REPO / "experiments" / "retrieval_s3_gate" / "analyze.py")
helpers = load("s3_pilot_test_helpers_for_gate", REPO / "tests" / "test_retrieval_s3_pilot.py")
requires_scifact = pytest.mark.skipif(not (DEFAULT_DATA_DIR / "scifact" / "corpus.jsonl").is_file(),
                                      reason="SciFact not downloaded")


def row(stratum="within", **kw) -> dict:
    base = {"stratum": stratum, "relative_gap": 0.5, "raw_gap": 1.0, "top_score": 10.0,
            "s1": 0.5, "s1_usd": 0.001, "s1_s": 4.0, "s3": 0.7, "s3_usd": 0.004, "s3_s": 10.0,
            "r1": 0.6, "r1_usd": 0.003, "r1_s": 7.0}
    return {**base, **kw}


def test_gated_takes_the_base_pipeline_only_when_unsure() -> None:
    r = row(relative_gap=0.2)
    assert gate.gated(r, "s3", "relative_gap", 0.3) == {"ndcg": 0.7, "usd": 0.004, "s": 10.0, "searched": 1.0}
    assert gate.gated(r, "s3", "relative_gap", 0.2)["ndcg"] == 0.5  # not below tau: S1
    assert gate.gated(r, "r1", "relative_gap", math.inf)["ndcg"] == 0.6  # rate 1.0: always search
    assert gate.gated(r, "r1", "relative_gap", -math.inf)["ndcg"] == 0.5  # never search


def test_weighted_reweights_stratum_means_to_population_shares() -> None:
    rows = [row("beyond", s1=1.0), row("within", s1=0.0), row("within", s1=0.0)]
    assert gate.weighted(rows, {"beyond": 1, "within": 3}, lambda r: r["s1"]) == pytest.approx(0.25)


def test_selection_rule_and_its_tie_breaks() -> None:
    def c(signal, rate, ndcg):
        return {"signal": signal, "achieved_rate_tune": rate, "ndcg": ndcg}
    assert gate.select([c("raw_gap", 0.5, 0.80), c("top_score", 0.9, 0.81)])["signal"] == "top_score"
    assert gate.select([c("raw_gap", 0.9, 0.81), c("top_score", 0.5, 0.81)])["achieved_rate_tune"] == 0.5
    assert gate.select([c("top_score", 0.5, 0.81), c("relative_gap", 0.5, 0.81)])["signal"] == "relative_gap"


@pytest.mark.parametrize("d1, d2, outcome", [
    ({"low": 0.01}, {"low": -0.015}, "gate removes the harm (not blind)"),
    ({"low": 0.01}, {"low": -0.0201}, "gate does not help"),
    ({"low": -0.001}, {"low": 0.0}, "gate loses the gain"),
    (None, {"low": 0.0}, None),
])
def test_check_outcome_rule(d1, d2, outcome) -> None:
    assert gate.check_outcome(d1, d2) == outcome


def synthetic(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    """Records on the pilot tests' synthetic data: S1 = BM25 order; S3 and R1 put
    the hidden relevant document of odd ("beyond") queries first."""
    data = helpers.make_dataset(tmp_path / "data")
    split = load_scifact(data, "train")
    halves = dev_partition(sorted(split.queries))
    index = gate.BM25Index(split.corpus)
    files = {n: tmp_path / f"{n}.jsonl" for n in gate.INPUTS}
    files["s3_pilot_manifest"] = tmp_path / "manifest.json"

    def bm25(q):
        return [d for d, _ in index.rank(split.queries[q])]

    def s1(q):
        top = bm25(q)[:20]
        return {"pass": 1, "condition": "plain", "query_id": q, "bm25_top": top, "ranking_top": top,
                "estimated_usd": 0.0013, "latency_s": 4.0}

    def episode(q):
        beyond = int(q[1:]) % 2 == 1
        ranking = ([f"z{int(q[1:]):02d}"] if beyond else []) + [d for d in bm25(q) if not d.startswith("z")]
        return {"query_id": q, "phase": "main", "stratum": "beyond" if beyond else "within", "ranking": ranking,
                "call_usd": {"agent": [0.0014], "rerank": [0.0016]}, "wall_s": 9.0}

    def write(name, rows):
        files[name].write_text("".join(json.dumps(r) + "\n" for r in rows))

    write("s1_pilot", [s1(q) for q in halves["tune"]])
    write("s1_tau", [])
    write("s1_check", [s1(q) for q in halves["check"]])
    write("s3_pilot", [episode(q) for q in halves["tune"]] + [{**episode(halves["tune"][0]), "phase": "smoke"}])
    write("r1_pilot", [episode(q) for q in halves["tune"]])
    write("s3_check", [episode(q) for q in halves["check"]])
    beyond = sum(int(q[1:]) % 2 for q in halves["tune"])
    files["s3_pilot_manifest"].write_text(json.dumps(
        {"strata_tune": {"beyond": beyond, "within_available": len(halves["tune"]) - beyond}}))
    return data, files


def test_end_to_end_on_synthetic_records(tmp_path: Path) -> None:
    data, files = synthetic(tmp_path)
    summary, rows = gate.analyze(data, files, pinned=False)
    for base in gate.BASES:
        curve = summary["part_a_curves"][base]
        assert len(curve) == len(gate.SIGNALS) * len(gate.RATES)
        always = [c for c in curve if c["target_rate"] == 1.0]
        assert all(c["achieved_rate_tune"] == 1.0 and c["tau"] == math.inf for c in always)
        assert summary["part_a_selected"][base]["achieved_rate_tune"] == 1.0  # searching helps every beyond query
    b = summary["part_b_check_not_blind"]
    assert b["queries"] == len(dev_partition(sorted(load_scifact(data, "train").queries))["check"])
    assert b["gated_minus_s3"]["mean"] == pytest.approx(0.0) and b["outcome"] is not None
    assert summary["queries"]["tune_sample"] == len([r for r in rows if r["half"] == "tune"])  # smoke excluded
    gate.write(tmp_path / "out", summary, rows)
    public = (tmp_path / "out" / "summary.json").read_text() + (tmp_path / "out" / "analysis.md").read_text()
    assert not re.search(r'"?q\d\d"?[ ,:]', public) and "Infinity" not in public and '"+inf"' in public


def test_pins_and_pairing_refusals(tmp_path: Path) -> None:
    data, files = synthetic(tmp_path)
    with pytest.raises(SystemExit, match="pinned SHA-256"):
        gate.analyze(data, files, pinned=True)
    lines = files["r1_pilot"].read_text().splitlines()
    files["r1_pilot"].write_text("".join(line + "\n" for line in lines[1:]))
    with pytest.raises(SystemExit, match="different queries"):
        gate.analyze(data, files, pinned=False)


@requires_scifact
def test_real_inputs_match_their_pins_and_reproduce_the_studies() -> None:
    summary, _ = gate.analyze(DEFAULT_DATA_DIR)
    assert summary["queries"] == {"tune_sample": 163, "tune_all_for_thresholds": 405, "check": 404}
    s3 = summary["part_a_selected"]["s3"]
    assert s3["always_search"]["ndcg"] == pytest.approx(0.809, abs=5e-4)  # the S3 pilot, tune-weighted
    assert s3["s1_reference"]["ndcg"] == pytest.approx(0.756, abs=5e-4)
    assert summary["part_b_check_not_blind"]["means"]["s3"] == pytest.approx(0.819, abs=5e-4)  # the S3 check study
