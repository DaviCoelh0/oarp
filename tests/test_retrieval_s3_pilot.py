"""S3 pilot runner and analyzer with scripted models (no API key, no paid call).

A synthetic BEIR-format dataset gives both strata: query j is "topic<j>"; for
even j the relevant document shares that word (BM25 rank 1, "within"), for odd
j it shares none and sorts behind 40 other documents ("beyond" rank 20).
Skipped without the `retrieval` and `langgraph` extras."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

pytest.importorskip("bm25s")
pytest.importorskip("langgraph")

from oarp.core.types import ModelResponse, ToolCall, Usage  # noqa: E402
from oarp.envs.retrieval.data import DEFAULT_DATA_DIR, dev_partition, load_scifact  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "experiments" / "retrieval_s3_pilot"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"s3_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("run_pilot")
analyzer = load("analyze")
requires_scifact = pytest.mark.skipif(not (DEFAULT_DATA_DIR / "scifact" / "corpus.jsonl").is_file(),
                                      reason="SciFact not downloaded")


def make_dataset(root: Path) -> Path:
    folder = root / "scifact"
    (folder / "qrels").mkdir(parents=True)
    docs = [{"_id": f"c{i:02d}", "title": f"T{i}", "text": f"topic{i} common"} for i in range(40)]
    docs += [{"_id": f"z{j:02d}", "title": f"U{j}", "text": f"unrelated{j} rare"} for j in range(1, 24, 2)]
    (folder / "corpus.jsonl").write_text("".join(json.dumps(d) + "\n" for d in docs))
    (folder / "queries.jsonl").write_text("".join(json.dumps({"_id": f"q{j:02d}", "text": f"topic{j}"}) + "\n"
                                                  for j in range(24)))
    qrels = ["query-id\tcorpus-id\tscore"] + [f"q{j:02d}\t{'c' if j % 2 == 0 else 'z'}{j:02d}\t1" for j in range(24)]
    (folder / "qrels" / "train.tsv").write_text("\n".join(qrels) + "\n")
    return root


class RescuingModel:
    """Agent: searches "unrelated<j>" (which finds the hidden relevant document of
    odd queries), then ranks the handles in reverse; reranker: numbers reversed."""

    def __init__(self, fail_agent: Exception | None = None) -> None:
        self.fail_agent = fail_agent

    def complete(self, messages, tools):
        usage = Usage(12_000, 400)
        if not tools:
            k = len(re.findall(r"^\[(\d+)\]", messages[-1].content, re.M))
            return ModelResponse(text="RANKING: " + ", ".join(str(i) for i in range(k, 0, -1)), usage=usage,
                                 model=runner.MODEL)
        if self.fail_agent is not None:
            raise self.fail_agent
        if not any(m.role == "tool" for m in messages):
            j = re.search(r"Claim: topic(\d+)", messages[1].content).group(1)
            return ModelResponse(tool_calls=(ToolCall("s", "search", {"query": f"unrelated{j}"}),), usage=usage,
                                 model=runner.MODEL)
        handles = sorted({int(n) for m in messages for n in re.findall(r"^\[D(\d+)\]", m.content, re.M)})
        return ModelResponse(tool_calls=(ToolCall("f", "finish", {"answer": "RANKING: " + ", ".join(
            f"D{n}" for n in reversed(handles))}),), usage=usage, model=runner.MODEL)


class ModelRefused(RuntimeError):
    """Same class name as the adapter's decline, which is what the runner checks."""


def run(tmp_path: Path, model=None, **kw) -> tuple[dict, Path, Path]:
    data = make_dataset(tmp_path / "data")
    out = tmp_path / "out"
    manifest = runner.run_pilot(model or RescuingModel(), data_dir=data, out=out, paid=False,
                                model_settings={"model": "test"}, within_n=kw.pop("within_n", 3),
                                smoke_n=kw.pop("smoke_n", 2), log=lambda _: None, **kw)
    return manifest, data, out


def episodes(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "private" / "episodes.jsonl").read_text().splitlines()]


# --- sample ------------------------------------------------------------------------

def test_sample_takes_all_beyond_then_hash_ordered_within_and_a_disjoint_smoke_set() -> None:
    beyond = {f"q{i}": i % 3 == 0 for i in range(30)}
    sample = runner.select_sample(beyond, within_n=5, smoke_n=2)
    inside = runner.study.hash_order(runner.ORDER_SEED, [q for q, b in beyond.items() if not b])
    assert {q for q in sample["main"] if beyond[q]} == {q for q, b in beyond.items() if b}
    assert set(sample["main"]) - {q for q, b in beyond.items() if b} == set(inside[:5])
    assert sample["smoke"] == inside[5:7] and not set(sample["smoke"]) & set(sample["main"])
    assert sample["main"] == runner.study.hash_order(runner.ORDER_SEED, sample["main"])  # one interleaved order
    assert runner.select_sample(dict(reversed(list(beyond.items()))), 5, 2) == sample


# --- runner ------------------------------------------------------------------------

def test_dry_run_runs_smoke_then_main_on_tune_only(tmp_path: Path) -> None:
    manifest, data, out = run(tmp_path)
    tune = set(dev_partition(sorted(load_scifact(data, "train").queries))["tune"])
    beyond = sum(1 for q in tune if int(q[1:]) % 2 == 1)
    assert manifest["strata_tune"] == {"beyond": beyond, "within_available": len(tune) - beyond}
    assert manifest["smoke_passed"] is True and not manifest["stopped"]
    assert manifest["completed_episodes"] == {"smoke": 2, "main": beyond + 3}
    rows = episodes(out)
    assert [r["phase"] for r in rows] == ["smoke"] * 2 + ["main"] * (beyond + 3)
    assert {r["query_id"] for r in rows} <= tune
    assert all(r["config"]["final_order"] == "forced_rerank" and r["rerank"] is not None for r in rows)
    public = (out / "manifest.json").read_text()
    assert not re.search(r'"q\d\d"', public)


def test_unclean_smoke_episode_stops_before_the_main_phase(tmp_path: Path) -> None:
    manifest, _, out = run(tmp_path, RescuingModel(fail_agent=RuntimeError("bad request")))
    assert manifest["smoke_passed"] is False and manifest["completed_episodes"] == {"smoke": 1, "main": 0}
    assert manifest["stop_reason"].startswith("smoke episode not clean")
    assert [r["smoke_ok"] for r in episodes(out)] == [False]


def test_a_declined_smoke_episode_is_still_clean(tmp_path: Path) -> None:
    manifest, _, _ = run(tmp_path, RescuingModel(fail_agent=ModelRefused("declined")))
    assert manifest["smoke_passed"] is True and manifest["completed_episodes"]["main"] > 0


def test_harness_error_stops_the_run_and_is_recorded(tmp_path: Path, monkeypatch) -> None:
    def broken(*args, **kwargs):
        raise KeyError("handles")
    monkeypatch.setattr(runner.agent_mod, "run_episode", broken)
    manifest, _, _ = run(tmp_path)
    assert manifest["harness_error"] == "KeyError: 'handles'" and manifest["stopped"]
    assert manifest["completed_episodes"] == {"smoke": 0, "main": 0} and manifest["smoke_passed"] is False


def test_budget_guard_stops_between_episodes_never_inside_one(tmp_path: Path) -> None:
    # Each scripted episode costs 3 x $0.0014 = $0.0042, so this cap admits exactly 3 episodes
    # (the 4th would need 0.0126 + 0.0495 > 0.0595), though one costs far less than the worst case.
    budget = runner.EPISODE_WORST_USD + 0.01
    manifest, _, out = run(tmp_path, within_n=6, budget_usd=budget)
    done = manifest["completed_episodes"]["smoke"] + manifest["completed_episodes"]["main"]
    assert manifest["stopped"] and "would exceed" in manifest["stop_reason"] and done == 3
    assert len(episodes(out)) == done and all(r["ranking"] for r in episodes(out))
    assert manifest["estimated_spend_usd"] + runner.EPISODE_WORST_USD > budget


def test_pins_refuse_a_changed_config_or_prompt(monkeypatch) -> None:
    runner.check_pins()  # the committed settings pass
    monkeypatch.setattr(runner, "CONFIG_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="AgentConfig"):
        runner.check_pins()
    monkeypatch.undo()
    monkeypatch.setattr(runner.agent_mod, "AGENT_INSTRUCTION", runner.agent_mod.AGENT_INSTRUCTION + " ")
    with pytest.raises(SystemExit, match="agent prompts"):
        runner.check_pins()


def test_stratum_file_must_match_its_hash_and_the_fresh_strata(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "per_query.jsonl"
    path.write_text(json.dumps({"query_id": "a", "beyond": True}) + "\n")
    with pytest.raises(SystemExit, match="pinned SHA-256"):
        runner.check_strata_file({"a": True}, path)
    monkeypatch.setattr(runner, "STRATA_SHA256", runner.study.sha256_file(path))
    assert runner.check_strata_file({"a": True}, path) == runner.STRATA_SHA256
    with pytest.raises(SystemExit, match="differ from freshly computed"):
        runner.check_strata_file({"a": False}, path)


@requires_scifact
def test_real_strata_match_the_pinned_file_and_the_contract() -> None:
    split = load_scifact(DEFAULT_DATA_DIR, "train")
    index = runner.BM25Index(split.corpus)
    beyond = runner.strata(split, index, dev_partition(sorted(split.queries))["tune"], 20)
    assert runner.check_strata_file(beyond) == runner.STRATA_SHA256
    assert sum(beyond.values()) == 63 and len(beyond) - sum(beyond.values()) == 342
    sample = runner.select_sample(beyond)
    assert len(sample["main"]) == 163 and len(sample["smoke"]) == 5


# --- analyzer ------------------------------------------------------------------------

def fake_s1(tmp_path: Path, data: Path) -> tuple[Path, Path]:
    """S1 records for every tune query, with S1 = BM25 order (so S1 = S0)."""
    split = load_scifact(data, "train")
    index = runner.BM25Index(split.corpus)
    pilot, tau = tmp_path / "pilot", tmp_path / "tau"
    for folder in (pilot, tau):
        (folder / "private").mkdir(parents=True)
        (folder / "manifest.json").write_text("{}")
        (folder / "private" / "sample.json").write_text("{}")
    rows = []
    for q in dev_partition(sorted(split.queries))["tune"]:
        top = [d for d, _ in index.rank(split.queries[q], 20)]
        rows.append({"pass": 1, "condition": "plain", "query_id": q, "bm25_top": top, "ranking_top": top,
                     "estimated_usd": 0.0013, "latency_s": 4.0})
    (pilot / "private" / "calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tau / "private" / "calls.jsonl").write_text("")
    return pilot, tau


def test_analyzer_scores_main_episodes_and_finds_the_rescues(tmp_path: Path) -> None:
    manifest, data, out = run(tmp_path)
    pilot, tau = fake_s1(tmp_path, data)
    summary = analyzer.main(["--results", str(out), "--pilot-results", str(pilot), "--tau-results", str(tau),
                             "--data-dir", str(data)])
    assert summary["episodes"]["scored"] == manifest["completed_episodes"]["main"]  # smoke excluded
    m = summary["mechanism"]
    assert m["rescue_rate_beyond"] == 1.0 and m["landing_rate_forced_rerank"] == 1.0
    assert summary["means"]["beyond"]["s1"] == 0.0 and summary["means"]["beyond"]["s3fr"] == 1.0
    assert summary["H2_beyond_s3fr_minus_s1"]["mean"] == 1.0
    assert summary["outcome_rules"]["run"] == "dry run: no outcome"
    assert summary["behaviour"]["beyond"]["mean_searches"] == 1.0
    public = (out / "summary.json").read_text() + (out / "analysis.md").read_text()
    assert not re.search(r'"?q\d\d"?[ ,:]', public)


def test_analyzer_refuses_episodes_without_an_s1_record(tmp_path: Path) -> None:
    _, data, out = run(tmp_path)
    pilot, tau = fake_s1(tmp_path, data)
    rows = (pilot / "private" / "calls.jsonl").read_text().splitlines()
    first_main = next(r["query_id"] for r in episodes(out) if r["phase"] == "main")
    (pilot / "private" / "calls.jsonl").write_text(
        "".join(line + "\n" for line in rows if json.loads(line)["query_id"] != first_main))
    with pytest.raises(SystemExit, match="no reused S1 record"):
        analyzer.analyze(out, pilot, tau, data)


def ci(mean: float, low: float, high: float) -> dict:
    return {"mean": mean, "low": low, "high": high}


VALID = {"scheduled_episodes": {"main": 163}, "completed_episodes": {"main": 163}, "harness_error": None,
         "smoke_passed": True, "stopped": False}


@pytest.mark.parametrize("h1, h2, h3, earns", [
    (ci(0.02, 0.005, 0.04), None, None, True),  # H1 positive
    (ci(0.01, -0.01, 0.03), ci(0.1, 0.05, 0.15), ci(-0.01, -0.019, 0.0), True),  # H2, H3, positive point
    (ci(0.01, -0.01, 0.03), ci(0.1, 0.05, 0.15), ci(-0.02, -0.03, 0.0), False),  # H3 fails
    (ci(-0.001, -0.02, 0.02), ci(0.1, 0.05, 0.15), ci(-0.01, -0.019, 0.0), False),  # point estimate <= 0
    (ci(0.01, -0.01, 0.03), ci(0.02, -0.01, 0.05), ci(0.0, -0.01, 0.01), False),  # H2 fails
])
def test_next_step_rule(h1, h2, h3, earns) -> None:
    rules = analyzer.classify(paid=True, manifest=VALID, other_error_rate=0.0, decline_rate=0.0, tool_bugs=0,
                              h1=h1, h2=h2, h3=h3)
    assert rules["run"] == "valid" and rules["earns_confirmation"] is earns


@pytest.mark.parametrize("change, run_outcome", [
    ({"harness_error": "KeyError"}, "invalid (harness error)"),
    ({"smoke_passed": False, "completed_episodes": {"main": 0}}, "stopped after the smoke phase (no result claimed)"),
    ({"stopped": True, "completed_episodes": {"main": 90}}, "incomplete (no hypothesis claimed)"),
])
def test_outcome_rules_for_unusable_runs(change, run_outcome) -> None:
    rules = analyzer.classify(paid=True, manifest={**VALID, **change}, other_error_rate=0.0, decline_rate=0.0,
                              tool_bugs=0, h1=ci(0.1, 0.05, 0.2), h2=None, h3=None)
    assert rules["run"] == run_outcome and rules["H1"] is None and rules["earns_confirmation"] is None


def test_outcome_rules_for_errors_and_declines() -> None:
    kw = dict(paid=True, manifest=VALID, h1=ci(0.1, 0.05, 0.2), h2=None, h3=None)
    assert analyzer.classify(other_error_rate=0.06, decline_rate=0.0, tool_bugs=0, **kw)["run"].startswith("invalid")
    assert analyzer.classify(other_error_rate=0.0, decline_rate=0.0, tool_bugs=1, **kw)["run"].startswith("invalid")
    rules = analyzer.classify(other_error_rate=0.0, decline_rate=0.11, tool_bugs=0, **kw)
    assert rules["run"] == "valid" and rules["decline_warning"] and rules["H1"] == "positive"
