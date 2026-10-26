"""Offline configuration search: routing, scoring, the selection rule, the
Pareto frontier, the space, pins and privacy; with the downloaded data, the
real pinned inputs. Skipped without the `retrieval` extra."""

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


cs = load("config_search", REPO / "experiments" / "retrieval_config_search" / "search.py")
gate_tests = load("gate_tests_for_search", REPO / "tests" / "test_retrieval_s3_gate.py")
requires_scifact = pytest.mark.skipif(not (DEFAULT_DATA_DIR / "scifact" / "corpus.jsonl").is_file(),
                                      reason="SciFact not downloaded")


def policy(st=-math.inf, rt=-math.inf, ss="relative_gap", rs="relative_gap", sr=0.0, rr=0.0) -> dict:
    return {"search_signal": ss, "search_tau": st, "search_rate": sr, "rerank_signal": rs, "rerank_tau": rt,
            "rerank_rate": rr}


def row(stratum="within", **kw) -> dict:
    base = {"stratum": stratum, "relative_gap": 0.5, "raw_gap": 1.0, "top_score": 10.0,
            "s0": 0.4, "s0_usd": 0.0, "s0_s": 0.0, "s1": 0.5, "s1_usd": 0.001, "s1_s": 4.0,
            "s3": 0.7, "s3_usd": 0.004, "s3_s": 10.0}
    return {**base, **kw}


def test_route_searches_first_then_reranks_then_keeps_bm25() -> None:
    r = row(top_score=5.0, raw_gap=2.0)
    assert cs.route(r, policy(st=6.0, ss="top_score")) == "s3"
    assert cs.route(r, policy(st=4.0, ss="top_score", rt=3.0, rs="raw_gap")) == "s1"
    assert cs.route(r, policy(st=4.0, ss="top_score", rt=1.0, rs="raw_gap")) == "s0"
    assert cs.route(r, policy(st=math.inf)) == "s3" and cs.route(r, policy()) == "s0"


def test_score_plain_and_reweighted() -> None:
    rows = [row("beyond", top_score=1.0), row("within", top_score=20.0), row("within", top_score=20.0)]
    p = policy(st=5.0, ss="top_score")  # only the beyond row searches
    plain = cs.score(rows, p)
    assert plain["ndcg"] == pytest.approx((0.7 + 0.4 + 0.4) / 3) and plain["share_s3"] == pytest.approx(1 / 3)
    weighted = cs.score(rows, p, {"beyond": 1, "within": 3})
    assert weighted["ndcg"] == pytest.approx(0.25 * 0.7 + 0.75 * 0.4) and weighted["usd"] == pytest.approx(0.001)


def test_selection_takes_the_cheapest_policy_within_tolerance() -> None:
    def r(ndcg, usd, sr=0.5, rr=0.5, ss="relative_gap", rs="relative_gap"):
        return {"ndcg": ndcg, "usd": usd, "search_rate": sr, "rerank_rate": rr, "search_signal": ss, "rerank_signal": rs}
    results = [r(0.820, 0.004), r(0.816, 0.002), r(0.814, 0.001), r(0.816, 0.002, sr=0.3)]
    best, selected = cs.select(results, tolerance=0.005)
    assert best["ndcg"] == 0.820 and (selected["usd"], selected["search_rate"]) == (0.002, 0.3)
    assert cs.select([r(0.82, 0.004), r(0.81, 0.001)], tolerance=0.005)[1]["usd"] == 0.004


def test_pareto_keeps_only_undominated_policies() -> None:
    results = [{"ndcg": 0.6, "usd": 0.0}, {"ndcg": 0.7, "usd": 0.001}, {"ndcg": 0.65, "usd": 0.002},
               {"ndcg": 0.8, "usd": 0.003}, {"ndcg": 0.8, "usd": 0.004}]
    assert [(p["ndcg"], p["usd"]) for p in cs.pareto(results)] == [(0.6, 0.0), (0.7, 0.001), (0.8, 0.003)]


def test_space_has_every_signal_and_rate_with_label_free_thresholds() -> None:
    values = {s: [float(i) for i in range(10)] for s in cs.SIGNALS}
    policies = cs.space(values)
    assert len(policies) == (len(cs.SIGNALS) * len(cs.RATES)) ** 2 == 1089
    rate0 = [p for p in policies if p["search_rate"] == 0.0]
    rate1 = [p for p in policies if p["search_rate"] == 1.0]
    assert all(p["search_tau"] == -math.inf for p in rate0) and all(p["search_tau"] == math.inf for p in rate1)


def test_end_to_end_on_synthetic_records(tmp_path: Path) -> None:
    data, gate_files = gate_tests.synthetic(tmp_path)
    halves = dev_partition(sorted(load_scifact(data, "train").queries))
    files = {n: gate_files[n] for n in cs.INPUTS}
    for name in ("s3_check", "s3_pilot"):  # the search also checks each episode's first 20 candidates
        split = load_scifact(data, "train")
        index = cs.BM25Index(split.corpus)
        rows = [json.loads(line) for line in files[name].read_text().splitlines()]
        for r in rows:
            r["candidates"] = [d for d, _ in index.rank(split.queries[r["query_id"]], 20)]
        files[name].write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary, rows = cs.analyze(data, files, pinned=False)
    assert summary["policies"] == 1089 and summary["queries"]["selection_check"] == len(halves["check"])
    assert summary["fixed_pipelines_check"]["S3"]["share_s3"] == 1.0
    assert summary["selected"]["ndcg"] >= summary["best_check"]["ndcg"] - cs.TOLERANCE - 1e-12
    cs.write(tmp_path / "out", summary, rows)
    public = (tmp_path / "out" / "summary.json").read_text() + (tmp_path / "out" / "analysis.md").read_text()
    assert not re.search(r'"?q\d\d"?[ ,:]', public) and "Infinity" not in public
    with pytest.raises(SystemExit, match="pinned SHA-256"):
        cs.analyze(data, files, pinned=True)


@requires_scifact
def test_real_inputs_reproduce_the_recorded_studies() -> None:
    summary, _ = cs.analyze(DEFAULT_DATA_DIR)
    fixed = summary["fixed_pipelines_check"]
    assert fixed["S0"]["ndcg"] == pytest.approx(0.666, abs=5e-4)
    assert fixed["S1"]["ndcg"] == pytest.approx(0.786, abs=5e-4)
    assert fixed["S3"]["ndcg"] == pytest.approx(0.819, abs=5e-4)
    assert summary["queries"] == {"selection_check": 404, "thresholds_dev": 809, "cross_check_tune": 163}
