"""R1 pilot, paid part: "reformulate once, then rerank" on the S3 pilot's 163
tune queries, paired with their S3 episodes; 5 smoke episodes first.

    venv/bin/python experiments/retrieval_s3_r1_pilot/run_pilot.py --dry-run   # free: scripted model, temp folder
    venv/bin/python experiments/retrieval_s3_r1_pilot/run_pilot.py             # PAID

Implements the R1 proposal's pilot (docs: R1 proposal, section 4; its
contract needs its own approval before a paid run):
- queries: exactly the S3 pilot's main sample (pinned sample file, with its
  strata), in SHA-256 order of ORDER_SEED + id;
- smoke: 5 tune queries within BM25 rank 20 used by neither the S3 pilot
  (main or smoke) nor the S3 check study's smoke phase, first in the same
  hash order; excluded from all analysis;
- config: the S3 pilot's AgentConfig with extra_search="once" (pinned hash);
- the S3 pilot runner is reused read-only by import (pinned SHA-256): its S3
  pins, strata, smoke criterion and stop helper; the guard allows an episode
  only if 2 worst-case calls still fit under the cap.

Refuses unless the S3 and R1 pins hold, and (paid) the S3 pilot runner, the
S3 pilot sample and the check study's sample match their pinned SHA-256, the
sample is 63 + 100 queries, the code is committed, nothing is untracked under
the guarded folders, results/ is empty and the dataset matches its pinned
SHA-256. Only the train split is loaded; only tune queries run.

Outputs: results/manifest.json (public; no query ids) and
results/private/episodes.jsonl (one record per episode; gitignored).
"""

import argparse
import importlib.util
import json
import re
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from oarp.core.env import read_env_value  # noqa: E402
from oarp.core.model import ModelClient  # noqa: E402
from oarp.core.pricing import PRICES_AS_OF  # noqa: E402
from oarp.core.types import Message, ModelResponse, ToolSpec, Usage  # noqa: E402
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
CHECK_SAMPLE = REPO / "experiments" / "retrieval_s3_check" / "results" / "private" / "sample.json"
CHECK_SAMPLE_SHA256 = "ffcf529b5733a770ace96e7015fc3b444a41d1906bb68756f2c9ca9479c2e0fa"
R1_CONFIG_SHA256 = "1d1a99b56ff19aa8a5b972b74f603976729c44034d1121ba4cfc1583b3896648"
R1_PROMPT_SHA256 = {
    "agent_system": "32a63607a221591460a98d61b83c5fa704104d363e34e53e7530584c72bb83dd",
    "reformulate_instruction": "30af968579708aec4fb0dd80ca13720eedb8fb2551531ec1ea45044f224b0272",
}
ORDER_SEED = "oarp-r2-s3-r1-pilot-v1:"
EXPECTED = {"main": 163, "beyond": 63}  # checked in paid mode
SMOKE_N = 5
BUDGET_USD = 0.80
GUARDED = ("src/", "tests/", "experiments/retrieval_s3_r1_pilot/")


def load_pilot_runner():
    spec = importlib.util.spec_from_file_location("s3_pilot_runner_for_r1", PILOT_DIR / "run_pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot = load_pilot_runner()
CONFIG = replace(pilot.CONFIG, extra_search="once")
EPISODE_WORST_USD = 2 * pilot.WORST_CASE_CALL_USD  # one reformulation call + one rerank call


def check_pins() -> None:
    pilot.check_pins()  # the S3 side (config, agent prompts, rerank prompts vs the S1 pilot) is unchanged
    if agent_mod.config_sha256(CONFIG) != R1_CONFIG_SHA256:
        raise SystemExit("refusing: the R1 AgentConfig differs from the pinned one")
    if agent_mod.r1_prompt_sha256() != R1_PROMPT_SHA256:
        raise SystemExit("refusing: the R1 prompts differ from the pinned ones")


def verify_pinned(*, paid: bool, pilot_sample: Path = PILOT_SAMPLE, check_sample: Path = CHECK_SAMPLE) -> dict:
    files = {"pilot_runner": PILOT_DIR / "run_pilot.py", "pilot_sample": pilot_sample, "check_sample": check_sample}
    missing = [n for n, p in files.items() if not p.is_file()]
    if missing:
        raise SystemExit(f"refusing: reused files missing: {missing}")
    hashes = {n: study.sha256_file(p) for n, p in files.items()}
    pins = {"pilot_runner": PILOT_RUNNER_SHA256, "pilot_sample": PILOT_SAMPLE_SHA256,
            "check_sample": CHECK_SAMPLE_SHA256}
    if paid:
        changed = [n for n in files if hashes[n] != pins[n]]
        if changed:
            raise SystemExit(f"refusing: reused files differ from their pinned SHA-256: {changed}")
    return hashes


def select(pilot_used: dict, check_smoke: Sequence[str], tune_beyond: dict[str, bool],
           smoke_n: int = SMOKE_N) -> dict[str, list[str]]:
    """Main: the S3 pilot's main sample in hash order. Smoke: spare tune within-20 queries."""
    taken = set(pilot_used["main"]) | set(pilot_used["smoke"]) | set(check_smoke)
    spare = [q for q, b in tune_beyond.items() if not b and q not in taken]
    return {"main": study.hash_order(ORDER_SEED, pilot_used["main"]),
            "smoke": study.hash_order(ORDER_SEED, spare)[:smoke_n]}


class DryRunModel:
    """Free stand-in. Reformulation: QUERY = the claim's last 3 words. Rerank:
    the shown numbers reversed."""

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelResponse:
        usage = Usage(12_000, 300)
        last = messages[-1].content
        if last.endswith(agent_mod.REFORMULATE_INSTRUCTION):
            claim = last.splitlines()[0].removeprefix("Claim: ")
            return ModelResponse(text="QUERY: " + " ".join(claim.split()[-3:]), usage=usage, model=pilot.MODEL)
        k = len(re.findall(r"^\[(\d+)\]", last, re.M))
        return ModelResponse(text="RANKING: " + ", ".join(str(i) for i in range(k, 0, -1)), usage=usage,
                             model=pilot.MODEL)


def run_r1(model: ModelClient, *, data_dir: Path, out: Path, paid: bool, model_settings: dict,
           smoke_n: int = SMOKE_N, budget_usd: float = BUDGET_USD, pilot_sample: Path = PILOT_SAMPLE,
           check_sample: Path = CHECK_SAMPLE, log: Callable[[str], None] = print) -> dict:
    check_pins()
    reused = verify_pinned(paid=paid, pilot_sample=pilot_sample, check_sample=check_sample)
    checks = study.preflight(REPO, out, data_dir, SCIFACT_SHA256, paid=paid, guarded=GUARDED)
    split = load_scifact(data_dir, "train")  # never the held-out split
    index = BM25Index(split.corpus, BM25Config())
    tune = dev_partition(sorted(split.queries))["tune"]
    tune_beyond = pilot.strata(split, index, tune, CONFIG.first_k)
    used = json.loads(pilot_sample.read_text())
    check_smoke = json.loads(check_sample.read_text())["smoke"]
    if set(used["main"]) - set(tune):
        raise SystemExit("refusing: the S3 pilot sample is not inside the tune half")
    if set(used["beyond"]) != {q for q in used["main"] if tune_beyond[q]}:
        raise SystemExit("refusing: the S3 pilot's strata differ from freshly computed ones")
    sample = select(used, check_smoke, tune_beyond, smoke_n)
    counts = {"main": len(sample["main"]), "beyond": sum(tune_beyond[q] for q in sample["main"])}
    if paid and (counts != EXPECTED or len(sample["smoke"]) != smoke_n):
        raise SystemExit(f"refusing: expected {EXPECTED} and {smoke_n} smoke queries, found {counts}")
    graph = agent_mod.build_agent_graph(CONFIG, model, index, split.corpus)

    (out / "private").mkdir(parents=True, exist_ok=True)
    (out / "private" / "sample.json").write_text(json.dumps(sample, indent=2) + "\n")
    path = out / "private" / "episodes.jsonl"
    guard = study.BudgetGuard(budget_usd, EPISODE_WORST_USD)
    manifest = {
        "study": "R1 pilot: reformulate once, then rerank, paired with the S3 pilot's tune episodes",
        "mode": "paid" if paid else "dry-run (scripted model and usage; no API call)",
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": pilot.MODEL, "model_settings": model_settings, "prices_as_of": PRICES_AS_OF,
        "agent_config": asdict(CONFIG), "agent_config_sha256": agent_mod.config_sha256(CONFIG),
        "r1_prompt_sha256": agent_mod.r1_prompt_sha256(), "rerank_prompt_sha256": rerank_mod.prompt_sha256(),
        "r1_prompts": {"system": agent_mod.AGENT_SYSTEM, "instruction": agent_mod.REFORMULATE_INSTRUCTION},
        **bm25_provenance(BM25Config()),
        "versions": {pkg: pilot._version(pkg) for pkg in ("anthropic", "langgraph", "bm25s", "numpy")},
        "python": sys.version.split()[0], "dataset_sha256": SCIFACT_SHA256, "reused_sha256": reused,
        "split": "train (tune half: the S3 pilot's sample; smoke on spare tune queries)",
        "dev_partition_seed": DEV_PARTITION_SEED, "order_seed": ORDER_SEED, "strata_sample": counts,
        "smoke_n": smoke_n, "scheduled_episodes": {"smoke": len(sample["smoke"]), "main": len(sample["main"])},
        "sample_sha256": study.sha256_text("\n".join(sample["main"] + ["--"] + sample["smoke"])),
        "budget_usd": budget_usd, "worst_case_call_usd": pilot.WORST_CASE_CALL_USD,
        "episode_worst_usd": EPISODE_WORST_USD, **checks,
        "script_sha256": {n: study.sha256_file(HERE / n) for n in ("run_pilot.py", "analyze.py") if (HERE / n).is_file()},
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
            study.append_record(path, {"index": i, "phase": phase,
                                       "stratum": "beyond" if tune_beyond[qid] else "within", "wall_s": wall,
                                       "call_usd": costs, "smoke_ok": ok, **asdict(record)})
            manifest["completed_episodes"][phase] += 1
            manifest["estimated_spend_usd"] = round(guard.spent, 6)
            manifest["calls_above_worst_case"] = guard.calls_above_worst_case
            study.write_manifest(out, manifest)
            log(f"[{phase} {i:3}/{len(queries)}] reformulation={record.reformulation_status:11} "
                f"new={len(record.candidates) - CONFIG.first_k} ok={ok} | spent ${guard.spent:.4f}")
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
        out = args.results or Path(tempfile.mkdtemp(prefix="r1-pilot-dryrun-"))
        model, settings = DryRunModel(), {"model": "dry-run"}
    else:
        from oarp.providers.anthropic_model import AnthropicModel, AnthropicSettings

        out = args.results or RESULTS
        key = read_env_value(args.env_file, "ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit(f"ANTHROPIC_API_KEY not found in {args.env_file}")
        base = AnthropicSettings()
        model, settings = AnthropicModel(base, api_key=key), asdict(base)
    manifest = run_r1(model, data_dir=args.data_dir, out=out, paid=not args.dry_run, model_settings=settings)
    print(f"results: {out}")
    return manifest


if __name__ == "__main__":
    main()
