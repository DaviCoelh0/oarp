"""S3 confirmation, paid part: the search agent (S3 + forced rerank), frozen at
the pilot's settings, on every check-half query; 5 smoke episodes first.

    venv/bin/python experiments/retrieval_s3_check/run_check.py --dry-run   # free: scripted model, temp folder
    venv/bin/python experiments/retrieval_s3_check/run_check.py             # PAID

Implements the approved contract (docs: S3 check contract; A1, cap $2.20):
- queries: all 404 check queries, in SHA-256 order of ORDER_SEED + id; labels
  are read only to tag each record with its stratum for the analysis;
- smoke: 5 TUNE queries within BM25 rank 20 and outside the pilot's main and
  smoke sets (the pilot's pinned sample file), first in the same hash order;
  excluded from all analysis; the main phase starts only after a clean smoke
  phase (the pilot's criterion);
- the pilot runner (`experiments/retrieval_s3_pilot/run_pilot.py`) is reused
  read-only by import, pinned by SHA-256: its config and prompt pins, strata,
  smoke criterion, dry-run model and per-episode guard values.

Refuses unless the pilot's pins hold, and (paid) the pilot runner, the pilot
sample and the S1 check study's records match their pinned SHA-256, the strata
are 55 / 349, the code is committed, nothing is untracked under the guarded
folders, results/ is empty and the dataset matches its pinned SHA-256. Only
the train split is loaded.

Outputs: results/manifest.json (public; no query ids) and
results/private/episodes.jsonl (one record per episode; gitignored).
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
from oarp.core.pricing import PRICES_AS_OF  # noqa: E402
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
PILOT_DIR = REPO / "experiments" / "retrieval_s3_pilot"
PILOT_RUNNER_SHA256 = "47e4a8cfaf0eb627429618ee2de09e66d3a415f2662ac288bda8eab22526af06"
PILOT_SAMPLE = PILOT_DIR / "results" / "private" / "sample.json"
PILOT_SAMPLE_SHA256 = "d19b6102eb0b46911a13e6f94f62de94f3dc1d25f552f482b2c4cef6a4429280"
S1_CHECK = REPO / "experiments" / "retrieval_s1_check" / "results"
S1_CHECK_SHA256 = {"manifest.json": "d0905eb86fbf3d973aa3bab69df0a3531327aa97f763ce01c9400539637c0220",
                   "private/sample.json": "e13ab3132ea07c933c62fdfc0ea718aa7fb26bcb369d8673b260f43501fe6238",
                   "private/calls.jsonl": "f0ff07cba7194c208848fbd6036772f13f41ded5da342fcfb068378cf064cb19"}
ORDER_SEED = "oarp-r2-s3-check-v1:"
EXPECTED = {"queries": 404, "beyond": 55, "within": 349}  # checked in paid mode
SMOKE_N = 5
BUDGET_USD = 2.20
GUARDED = ("src/", "tests/", "experiments/retrieval_s3_check/")


def load_pilot_runner():
    spec = importlib.util.spec_from_file_location("s3_pilot_runner", PILOT_DIR / "run_pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot = load_pilot_runner()


def verify_pinned(*, paid: bool, pilot_sample: Path = PILOT_SAMPLE, s1_check: Path = S1_CHECK) -> dict:
    """Hashes of the reused files; in paid mode every one must match its pin."""
    files = {"pilot_runner": PILOT_DIR / "run_pilot.py", "pilot_sample": pilot_sample,
             **{f"s1_check/{f}": s1_check / f for f in S1_CHECK_SHA256}}
    missing = [n for n, p in files.items() if not p.is_file()]
    if missing:
        raise SystemExit(f"refusing: reused files missing: {missing}")
    hashes = {n: study.sha256_file(p) for n, p in files.items()}
    pins = {"pilot_runner": PILOT_RUNNER_SHA256, "pilot_sample": PILOT_SAMPLE_SHA256,
            **{f"s1_check/{f}": h for f, h in S1_CHECK_SHA256.items()}}
    if paid:
        changed = [n for n in files if hashes[n] != pins[n]]
        if changed:
            raise SystemExit(f"refusing: reused files differ from their pinned SHA-256: {changed}")
    return hashes


def select(check_beyond: dict[str, bool], tune_beyond: dict[str, bool], pilot_used: set[str],
           smoke_n: int = SMOKE_N) -> dict[str, list[str]]:
    """Main: every check query in hash order. Smoke: tune within-20 queries the
    pilot did not use, first `smoke_n` in the same hash order."""
    spare = [q for q, b in tune_beyond.items() if not b and q not in pilot_used]
    return {"main": study.hash_order(ORDER_SEED, list(check_beyond)),
            "smoke": study.hash_order(ORDER_SEED, spare)[:smoke_n]}


def run_check(model: ModelClient, *, data_dir: Path, out: Path, paid: bool, model_settings: dict,
              smoke_n: int = SMOKE_N, budget_usd: float = BUDGET_USD, pilot_sample: Path = PILOT_SAMPLE,
              s1_check: Path = S1_CHECK, log: Callable[[str], None] = print) -> dict:
    pilot.check_pins()
    reused = verify_pinned(paid=paid, pilot_sample=pilot_sample, s1_check=s1_check)
    checks = study.preflight(REPO, out, data_dir, SCIFACT_SHA256, paid=paid, guarded=GUARDED)
    split = load_scifact(data_dir, "train")  # never the held-out split
    index = BM25Index(split.corpus, BM25Config())
    halves = dev_partition(sorted(split.queries))
    depth = pilot.CONFIG.first_k
    check_beyond = pilot.strata(split, index, halves["check"], depth)
    tune_beyond = pilot.strata(split, index, halves["tune"], depth)
    counts = {"queries": len(check_beyond), "beyond": sum(check_beyond.values()),
              "within": len(check_beyond) - sum(check_beyond.values())}
    if paid and counts != EXPECTED:
        raise SystemExit(f"refusing: expected check strata {EXPECTED}, found {counts}")
    used = json.loads(pilot_sample.read_text())
    sample = select(check_beyond, tune_beyond, set(used["main"]) | set(used["smoke"]), smoke_n)
    if paid and len(sample["smoke"]) != smoke_n:
        raise SystemExit("refusing: not enough spare tune queries for the smoke phase")
    stratum = {q: "beyond" if b else "within" for q, b in {**tune_beyond, **check_beyond}.items()}
    graph = agent_mod.build_agent_graph(pilot.CONFIG, model, index, split.corpus)

    (out / "private").mkdir(parents=True, exist_ok=True)
    (out / "private" / "sample.json").write_text(json.dumps(sample, indent=2) + "\n")
    path = out / "private" / "episodes.jsonl"
    guard = study.BudgetGuard(budget_usd, pilot.EPISODE_WORST_USD)
    manifest = {
        "study": "S3 confirmation: search agent (S3 + forced rerank) on the check half, settings frozen at the pilot",
        "mode": "paid" if paid else "dry-run (scripted model and usage; no API call)",
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": pilot.MODEL, "model_settings": model_settings, "prices_as_of": PRICES_AS_OF,
        "agent_config": asdict(pilot.CONFIG), "agent_config_sha256": agent_mod.config_sha256(pilot.CONFIG),
        "agent_prompt_sha256": agent_mod.prompt_sha256(), "rerank_prompt_sha256": rerank_mod.prompt_sha256(),
        **bm25_provenance(BM25Config()),
        "versions": {pkg: pilot._version(pkg) for pkg in ("anthropic", "langgraph", "bm25s", "numpy")},
        "python": sys.version.split()[0], "dataset_sha256": SCIFACT_SHA256, "reused_sha256": reused,
        "split": "train (check half; smoke on spare tune queries)", "dev_partition_seed": DEV_PARTITION_SEED,
        "order_seed": ORDER_SEED, "strata_check": counts, "smoke_n": smoke_n,
        "scheduled_episodes": {"smoke": len(sample["smoke"]), "main": len(sample["main"])},
        "sample_sha256": study.sha256_text("\n".join(sample["main"] + ["--"] + sample["smoke"])),
        "budget_usd": budget_usd, "worst_case_call_usd": pilot.WORST_CASE_CALL_USD,
        "episode_worst_usd": pilot.EPISODE_WORST_USD, **checks,
        "script_sha256": {n: study.sha256_file(HERE / n) for n in ("run_check.py", "analyze.py") if (HERE / n).is_file()},
        "completed_episodes": {"smoke": 0, "main": 0}, "estimated_spend_usd": 0.0, "smoke_passed": None,
        "stopped": False, "stop_reason": None, "harness_error": None, "calls_above_worst_case": 0,
    }
    study.write_manifest(out, manifest)

    def run_phase(phase: str, queries: list[str]) -> bool:
        for i, qid in enumerate(queries, 1):
            if not guard.can_call():
                return pilot._stop(manifest, guard.stop_reason() or "budget", log)
            started = time.perf_counter()
            try:
                record = agent_mod.run_episode(graph, qid, split.queries[qid])
            except Exception as exc:  # any graph exception is a harness error: the run is invalid
                manifest["harness_error"] = f"{type(exc).__name__}: {exc}"
                return pilot._stop(manifest, f"harness error in a {phase} episode", log)
            wall = time.perf_counter() - started
            costs = agent_mod.call_costs(record)
            every = costs["agent"] + costs["rerank"]
            guard.add(sum(c for c in every if c is not None) + pilot.WORST_CASE_CALL_USD * sum(c is None for c in every))
            above = sum(c is not None and c > pilot.WORST_CASE_CALL_USD for c in every)
            guard.calls_above_worst_case += above
            ok = pilot.episode_ok(record)
            study.append_record(path, {"index": i, "phase": phase, "stratum": stratum[qid], "wall_s": wall,
                                       "call_usd": costs, "smoke_ok": ok, **asdict(record)})
            manifest["completed_episodes"][phase] += 1
            manifest["estimated_spend_usd"] = round(guard.spent, 6)
            manifest["calls_above_worst_case"] = guard.calls_above_worst_case
            study.write_manifest(out, manifest)
            log(f"[{phase} {i:3}/{len(queries)}] stop={record.stop_reason:12} searches="
                f"{record.tool_calls['executed_searches']} ok={ok} | spent ${guard.spent:.4f}")
            if above:
                return pilot._stop(manifest, "a call cost more than the assumed worst case: the bound is unsafe", log)
            if phase == "smoke" and not ok:
                return pilot._stop(manifest, "smoke episode not clean (non-decline model error or tool bug)", log)
        return True

    manifest["smoke_passed"] = run_phase("smoke", sample["smoke"])
    study.write_manifest(out, manifest)
    if manifest["smoke_passed"]:  # the main phase never starts after an unclean smoke phase
        run_phase("main", sample["main"])
    manifest["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    study.write_manifest(out, manifest)
    return manifest


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="scripted model, no API calls, temp output")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results", type=Path, default=None, help="output folder (dry run: a temp folder)")
    parser.add_argument("--env-file", type=Path, default=REPO / ".env")
    args = parser.parse_args(argv)

    if args.dry_run:
        out = args.results or Path(tempfile.mkdtemp(prefix="s3-check-dryrun-"))
        model, settings = pilot.DryRunModel(), {"model": "dry-run"}
    else:
        from oarp.providers.anthropic_model import AnthropicModel, AnthropicSettings

        out = args.results or RESULTS
        key = read_env_value(args.env_file, "ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit(f"ANTHROPIC_API_KEY not found in {args.env_file}")
        base = AnthropicSettings()
        model, settings = AnthropicModel(base, api_key=key), asdict(base)
    manifest = run_check(model, data_dir=args.data_dir, out=out, paid=not args.dry_run, model_settings=settings)
    print(f"results: {out}")
    return manifest


if __name__ == "__main__":
    main()
