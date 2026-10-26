"""Final held-out evaluation, paid part: S1 (one rerank call) and S3 (search
agent + forced rerank) on every SciFact TEST query; 5 smoke queries first.

    venv/bin/python experiments/retrieval_heldout/run_test.py --dry-run   # free: scripted model, DEV stand-in
    venv/bin/python experiments/retrieval_heldout/run_test.py             # PAID: loads the test split

Implements the approved contract (docs: held-out contract; cap $2.30). The
test split is loaded ONLY in paid mode (`include_heldout=True`). A dry run on
the real data uses 300 development check queries (the test split's size) as a
stand-in, so no
test-split quantity is ever computed before the one-shot run; tests use a
synthetic test split.

- per query, in SHA-256 order of ORDER_SEED + id: S1 (`rerank`, S1-plain),
  then S3 (the S3 pilot's AgentConfig, forced rerank); one record per query;
- smoke: 5 tune within-20 queries used by no earlier study (the S3 pilot's
  main and smoke sets, the check and R1 smoke sets), excluded from analysis;
- guard: a query starts only if 10 worst-case calls (1 S1 + 9 S3) still fit
  under the cap; a call above the worst case stops the run; any graph
  exception is a harness error and stops the run;
- the S3 pilot runner is reused read-only by import (pinned): its S3 pins,
  strata, smoke criterion and stop helper.

Refuses unless the S3 pins hold, the reused sample files match their pinned
SHA-256 (paid), the test split has exactly 300 queries (paid), the code is
committed, nothing is untracked under the guarded folders, results/ is empty
and the dataset matches its pinned SHA-256 (paid).

Outputs: results/manifest.json (public; no query ids) and
results/private/records.jsonl (one record per query; gitignored).
"""

import argparse
import importlib.util
import json
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from oarp.core.env import read_env_value  # noqa: E402
from oarp.core.model import ModelClient  # noqa: E402
from oarp.core.pricing import PRICES_AS_OF, estimate_cost  # noqa: E402
from oarp.envs.retrieval import agent as agent_mod  # noqa: E402
from oarp.envs.retrieval import rerank as rerank_mod  # noqa: E402
from oarp.envs.retrieval import study  # noqa: E402
from oarp.envs.retrieval.bm25 import BM25Config, BM25Index  # noqa: E402
from oarp.envs.retrieval.bm25 import provenance as bm25_provenance  # noqa: E402
from oarp.envs.retrieval.data import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEV_PARTITION_SEED,
    SCIFACT_SHA256,
    dev_partition,
    load_scifact,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
EXP = REPO / "experiments"
PILOT_DIR = EXP / "retrieval_s3_pilot"
PILOT_RUNNER_SHA256 = "47e4a8cfaf0eb627429618ee2de09e66d3a415f2662ac288bda8eab22526af06"
USED_SAMPLES = {  # earlier studies' query sets on tune, so the smoke set is fresh
    "s3_pilot": EXP / "retrieval_s3_pilot" / "results" / "private" / "sample.json",
    "s3_check": EXP / "retrieval_s3_check" / "results" / "private" / "sample.json",
    "r1_pilot": EXP / "retrieval_s3_r1_pilot" / "results" / "private" / "sample.json",
}
USED_SAMPLES_SHA256 = {
    "s3_pilot": "d19b6102eb0b46911a13e6f94f62de94f3dc1d25f552f482b2c4cef6a4429280",
    "s3_check": "ffcf529b5733a770ace96e7015fc3b444a41d1906bb68756f2c9ca9479c2e0fa",
    "r1_pilot": "65cee92d7408544db368d62b414d428cd9666b0a76439a6c141371522b4cad5b",
}
S1_CONFIG = rerank_mod.RerankConfig()  # S1-plain, as confirmed
ORDER_SEED = "oarp-r2-heldout-v1:"
EXPECTED_TEST_QUERIES = 300  # BEIR's published SciFact test size; checked in paid mode
SMOKE_N = 5
BUDGET_USD = 2.30
GUARDED = ("src/", "tests/", "experiments/retrieval_heldout/")


def load_pilot_runner():
    spec = importlib.util.spec_from_file_location("s3_pilot_runner_for_heldout", PILOT_DIR / "run_pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot = load_pilot_runner()
QUERY_WORST_USD = (pilot.CONFIG.max_model_calls + 2) * pilot.WORST_CASE_CALL_USD  # S1 call + 8 agent + 1 rerank


def verify_pinned(*, paid: bool, samples: dict[str, Path] = USED_SAMPLES) -> dict[str, str]:
    files = {"pilot_runner": PILOT_DIR / "run_pilot.py", **samples}
    missing = [n for n, p in files.items() if not p.is_file()]
    if missing:
        raise SystemExit(f"refusing: reused files missing: {missing}")
    hashes = {n: study.sha256_file(p) for n, p in files.items()}
    pins = {"pilot_runner": PILOT_RUNNER_SHA256, **USED_SAMPLES_SHA256}
    if paid:
        changed = [n for n in files if hashes[n] != pins[n]]
        if changed:
            raise SystemExit(f"refusing: reused files differ from their pinned SHA-256: {changed}")
    return hashes


def smoke_queries(tune_beyond: dict[str, bool], used: set[str], n: int = SMOKE_N) -> list[str]:
    spare = [q for q, b in tune_beyond.items() if not b and q not in used]
    return study.hash_order(ORDER_SEED, spare)[:n]


def s1_record(model: ModelClient, qid: str, claim: str, ranking: list[str], corpus: dict[str, str]) -> dict:
    record = rerank_mod.rerank(model, qid, claim, ranking, corpus, S1_CONFIG)
    return {"bm25_top": ranking[:S1_CONFIG.k], "ranking_top": record.ranking[:S1_CONFIG.k], "status": record.status,
            "usage": asdict(record.usage) if record.usage else None,
            "estimated_usd": estimate_cost(pilot.MODEL, record.usage), "latency_s": record.latency_s,
            "model_reported": record.model, "error_kind": record.error_kind, "error": record.error,
            "text": record.text}


def s1_ok(rec: dict) -> bool:
    return not (rec["status"] == "model_error" and rec["error_kind"] != pilot.DECLINE)


def run_test(model: ModelClient, *, data_dir: Path, out: Path, paid: bool, model_settings: dict,
             query_split: str = "test", smoke_n: int = SMOKE_N, budget_usd: float = BUDGET_USD,
             samples: dict[str, Path] = USED_SAMPLES, log: Callable[[str], None] = print) -> dict:
    """query_split: "test" (the held-out split; required in paid mode) or
    "check-standin" (the development check half, for a dry run on real data)."""
    if query_split not in ("test", "check-standin"):
        raise ValueError("query_split must be 'test' or 'check-standin'")
    if paid and query_split != "test":
        raise SystemExit("refusing: a paid run evaluates the test split")
    pilot.check_pins()
    reused = verify_pinned(paid=paid, samples=samples)
    checks = study.preflight(REPO, out, data_dir, SCIFACT_SHA256, paid=paid, guarded=GUARDED)
    dev = load_scifact(data_dir, "train")
    index = BM25Index(dev.corpus, BM25Config())
    halves = dev_partition(sorted(dev.queries))
    tune_beyond = pilot.strata(dev, index, halves["tune"], pilot.CONFIG.first_k)
    used: set[str] = set()
    for path in samples.values():
        sample = json.loads(path.read_text())
        used |= set(sample.get("main", [])) | set(sample.get("smoke", []))
    smoke = smoke_queries(tune_beyond, used, smoke_n)
    if query_split == "test":
        held = load_scifact(data_dir, "test", include_heldout=True)  # the one-shot load
        if held.corpus != dev.corpus:
            raise SystemExit("refusing: the test split's corpus differs from the development corpus")
        claims = held.queries
    else:
        stand_in = study.hash_order(ORDER_SEED, halves["check"])[:EXPECTED_TEST_QUERIES]  # same size as test
        claims = {q: dev.queries[q] for q in stand_in}
    if paid and (len(claims) != EXPECTED_TEST_QUERIES or len(smoke) != smoke_n):
        raise SystemExit(f"refusing: expected {EXPECTED_TEST_QUERIES} test and {smoke_n} smoke queries, "
                         f"found {len(claims)} and {len(smoke)}")
    queries = study.hash_order(ORDER_SEED, list(claims))
    all_claims = {**{q: dev.queries[q] for q in smoke}, **claims}
    graph = agent_mod.build_agent_graph(pilot.CONFIG, model, index, dev.corpus)

    (out / "private").mkdir(parents=True, exist_ok=True)
    (out / "private" / "sample.json").write_text(json.dumps({"main": queries, "smoke": smoke}, indent=2) + "\n")
    path = out / "private" / "records.jsonl"
    guard = study.BudgetGuard(budget_usd, QUERY_WORST_USD)
    manifest = {
        "study": "Held-out evaluation: S1 and S3 on every SciFact test query (P* and S0 scored offline)",
        "mode": "paid" if paid else "dry-run (scripted model and usage; no API call)",
        "query_split": query_split, "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": pilot.MODEL, "model_settings": model_settings, "prices_as_of": PRICES_AS_OF,
        "s1_config": asdict(S1_CONFIG), "s3_config": asdict(pilot.CONFIG),
        "s3_config_sha256": agent_mod.config_sha256(pilot.CONFIG), "agent_prompt_sha256": agent_mod.prompt_sha256(),
        "rerank_prompt_sha256": rerank_mod.prompt_sha256(), **bm25_provenance(BM25Config()),
        "versions": {pkg: pilot._version(pkg) for pkg in ("anthropic", "langgraph", "bm25s", "numpy")},
        "python": sys.version.split()[0], "dataset_sha256": SCIFACT_SHA256, "reused_sha256": reused,
        "dev_partition_seed": DEV_PARTITION_SEED, "order_seed": ORDER_SEED,
        "scheduled": {"smoke": len(smoke), "main": len(queries)},
        "sample_sha256": study.sha256_text("\n".join(queries + ["--"] + smoke)),
        "budget_usd": budget_usd, "worst_case_call_usd": pilot.WORST_CASE_CALL_USD, "query_worst_usd": QUERY_WORST_USD,
        **checks,
        "script_sha256": {n: study.sha256_file(HERE / n) for n in ("run_test.py", "analyze.py") if (HERE / n).is_file()},
        "completed": {"smoke": 0, "main": 0}, "estimated_spend_usd": 0.0, "smoke_passed": None,
        "stopped": False, "stop_reason": None, "harness_error": None, "calls_above_worst_case": 0,
    }
    study.write_manifest(out, manifest)

    def run_phase(phase: str, qids: list[str]) -> bool:
        for i, qid in enumerate(qids, 1):
            if not guard.can_call():
                return pilot._stop(manifest, guard.stop_reason() or "budget", log)
            claim = all_claims[qid]
            ranking = [d for d, _ in index.rank(claim)]
            try:
                s1 = s1_record(model, qid, claim, ranking, dev.corpus)
                started = time.perf_counter()
                episode = agent_mod.run_episode(graph, qid, claim)
                wall = time.perf_counter() - started
            except Exception as exc:  # any exception here is a harness error: the run is invalid
                manifest["harness_error"] = f"{type(exc).__name__}: {exc}"
                return pilot._stop(manifest, f"harness error in a {phase} query", log)
            costs = agent_mod.call_costs(episode)
            every = [s1["estimated_usd"]] + costs["agent"] + costs["rerank"]
            guard.add(sum(c for c in every if c is not None) + pilot.WORST_CASE_CALL_USD * sum(c is None for c in every))
            above = sum(c is not None and c > pilot.WORST_CASE_CALL_USD for c in every)
            guard.calls_above_worst_case += above
            ok = s1_ok(s1) and pilot.episode_ok(episode)
            study.append_record(path, {"index": i, "phase": phase, "query_id": qid, "smoke_ok": ok, "s1": s1,
                                       "s3": {"wall_s": wall, "call_usd": costs, **asdict(episode)}})
            manifest["completed"][phase] += 1
            manifest["estimated_spend_usd"] = round(guard.spent, 6)
            manifest["calls_above_worst_case"] = guard.calls_above_worst_case
            study.write_manifest(out, manifest)
            log(f"[{phase} {i:3}/{len(qids)}] s1={s1['status']:11} s3={episode.stop_reason:12} ok={ok} "
                f"| spent ${guard.spent:.4f}")
            if above:
                return pilot._stop(manifest, "a call cost more than the assumed worst case: the bound is unsafe", log)
            if phase == "smoke" and not ok:
                return pilot._stop(manifest, "smoke query not clean (non-decline model error or tool bug)", log)
        return True

    manifest["smoke_passed"] = run_phase("smoke", smoke)
    study.write_manifest(out, manifest)
    if manifest["smoke_passed"]:  # the main phase never starts after an unclean smoke phase
        run_phase("main", queries)
    manifest["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    study.write_manifest(out, manifest)
    return manifest


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="scripted model, no API calls, temp output, DEV check queries as a stand-in")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results", type=Path, default=None, help="output folder (dry run: a temp folder)")
    parser.add_argument("--env-file", type=Path, default=REPO / ".env")
    args = parser.parse_args(argv)

    if args.dry_run:
        out = args.results or Path(tempfile.mkdtemp(prefix="heldout-dryrun-"))
        manifest = run_test(pilot.DryRunModel(), data_dir=args.data_dir, out=out, paid=False,
                            model_settings={"model": "dry-run"}, query_split="check-standin")
    else:
        from oarp.providers.anthropic_model import AnthropicModel, AnthropicSettings

        out = args.results or RESULTS
        key = read_env_value(args.env_file, "ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit(f"ANTHROPIC_API_KEY not found in {args.env_file}")
        base = AnthropicSettings()
        manifest = run_test(AnthropicModel(base, api_key=key), data_dir=args.data_dir, out=out, paid=True,
                            model_settings=asdict(base), query_split="test")
    print(f"results: {out}")
    return manifest


if __name__ == "__main__":
    main()
