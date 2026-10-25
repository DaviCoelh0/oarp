"""S3 pilot, paid part: the search agent (S3 + forced rerank) on a stratified
tune sample, first 5 smoke episodes, then the 163 sampled ones.

    venv/bin/python experiments/retrieval_s3_pilot/run_pilot.py --dry-run   # free: scripted model, temp folder
    venv/bin/python experiments/retrieval_s3_pilot/run_pilot.py             # PAID

Implements the approved contract (docs: S3 pilot contract; cap $1.60):
- sample: every tune query whose first relevant document is beyond BM25 rank
  20 (63) plus the first 100 of the others in SHA-256 order of ORDER_SEED + id;
  smoke: the next 5 of the others, excluded from all analysis; all episodes in
  one SHA-256 order of the same seed (strata interleaved). Labels are read only
  here, to form the strata; the agent never sees them;
- the smoke phase must be clean (no harness error, no non-decline model error,
  complete records) before the main phase starts;
- budget guard per episode: an episode starts only if the estimated spend plus
  EPISODE_WORST_USD (9 worst-case calls) stays within the cap; a call costing
  more than WORST_CASE_CALL_USD makes the bound unsafe and stops the run;
- a harness error (any exception from the graph) stops the run and is recorded.

Refuses unless the AgentConfig and the agent and rerank prompts equal their
pinned hashes, the rerank prompts equal the S1 pilot's (the reused S1 records
must come from the same reranker), and (paid) the stratum file matches its
pinned SHA-256 and the fresh strata, the code is committed, nothing is
untracked under the guarded folders, results/ is empty and the dataset matches
its pinned SHA-256. Only the train split is loaded; only tune queries run.

Outputs: results/manifest.json (public; no query ids) and
results/private/episodes.jsonl (one record per episode; gitignored).
"""

import argparse
import json
import re
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from oarp.core.env import read_env_value  # noqa: E402
from oarp.core.model import ModelClient  # noqa: E402
from oarp.core.pricing import PRICES_AS_OF  # noqa: E402
from oarp.core.types import Message, ModelResponse, ToolCall, ToolSpec, Usage  # noqa: E402
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
from oarp.envs.retrieval.metrics import first_relevant_rank  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
STRATA_FILE = REPO / "experiments" / "retrieval_s3_reachability" / "results" / "private" / "per_query.jsonl"
STRATA_SHA256 = "acd0a5932947a14900ea9e75a8f38cac1cf2e9c6d60f3b7af730114f06fa331a"
PILOT_MANIFEST = REPO / "experiments" / "retrieval_s1_pilot" / "results" / "manifest.json"
MODEL = "claude-haiku-5-5"
CONFIG = agent_mod.AgentConfig(final_order="forced_rerank")
CONFIG_SHA256 = "8d7a08fdcd7d69c25bb610e5ba92d7483320002c204ab131b25d72bb11b19418"
AGENT_PROMPT_SHA256 = {
    "agent_system": "32a63607a221591460a98d61b83c5fa704104d363e34e53e7530584c72bb83dd",
    "agent_instruction": "05c6e0636087326ffd5d17ea148cb36ec23323d0bafc738632ae7389b52869d8",
    "search_tool_description": "fec9cd71670a8db4ebd02d5a2481c051558b7cca5bcd2f1b0ad19356d88b8f49",
}
ORDER_SEED = "oarp-r2-s3-pilot-v1:"
EXPECTED = {"beyond": 63, "within_available": 342}  # checked in paid mode
WITHIN_N = 100
SMOKE_N = 5
BUDGET_USD = 1.60
WORST_CASE_CALL_USD = 0.0055  # ~30k input tokens (2.6 chars/token, replayed thinking) + 4,096 output
EPISODE_WORST_USD = (CONFIG.max_model_calls + 1) * WORST_CASE_CALL_USD
DECLINE = "ModelRefused"
GUARDED = ("src/", "tests/", "experiments/retrieval_s3_pilot/")


def check_pins(pilot_manifest: Path = PILOT_MANIFEST) -> None:
    if agent_mod.config_sha256(CONFIG) != CONFIG_SHA256:
        raise SystemExit("refusing: the AgentConfig differs from the pinned one")
    if agent_mod.prompt_sha256() != AGENT_PROMPT_SHA256:
        raise SystemExit("refusing: the agent prompts differ from the pinned ones")
    pilot = json.loads(pilot_manifest.read_text())
    if pilot.get("prompt_sha256") != rerank_mod.prompt_sha256():
        raise SystemExit("refusing: the rerank prompts differ from the S1 pilot's")
    if CONFIG.rerank_doc_chars != rerank_mod.RerankConfig().max_doc_chars or CONFIG.rerank_shuffle:
        raise SystemExit("refusing: the rerank settings differ from S1-plain")


def strata(split, index: BM25Index, tune: Sequence[str], depth: int) -> dict[str, bool]:
    """Query id -> True when BM25's first relevant document is beyond `depth` (reads labels)."""
    beyond = {}
    for q in tune:
        rank = first_relevant_rank([d for d, _ in index.rank(split.queries[q])], split.qrels[q])
        beyond[q] = rank is None or rank > depth
    return beyond


def check_strata_file(beyond: dict[str, bool], path: Path = STRATA_FILE) -> str:
    """Paid mode: the pinned reachability file must exist, match its hash and the fresh strata."""
    if not path.is_file():
        raise SystemExit(f"refusing: stratum file missing: {path}")
    sha = study.sha256_file(path)
    if sha != STRATA_SHA256:
        raise SystemExit("refusing: the stratum file differs from its pinned SHA-256")
    pinned = {r["query_id"]: r["beyond"] for r in map(json.loads, path.read_text().splitlines())}
    if pinned != beyond:
        raise SystemExit("refusing: the pinned strata differ from freshly computed ones")
    return sha


def select_sample(beyond: dict[str, bool], within_n: int = WITHIN_N, smoke_n: int = SMOKE_N) -> dict[str, list[str]]:
    """Beyond: all; within: the first `within_n` in hash order; smoke: the next
    `smoke_n`; run order: one hash order over the sample."""
    inside = study.hash_order(ORDER_SEED, [q for q, b in beyond.items() if not b])
    sample = [q for q, b in beyond.items() if b] + inside[:within_n]
    return {"main": study.hash_order(ORDER_SEED, sample), "smoke": inside[within_n:within_n + smoke_n]}


def episode_ok(record: agent_mod.EpisodeRecord) -> bool:
    """Smoke criterion: no non-decline model error in the agent or the rerank call, no tool bug."""
    agent_error = record.stop_reason == "model_error" and not (record.error or "").startswith(DECLINE)
    rr = record.rerank
    rerank_error = rr is not None and rr["status"] == "model_error" and rr["error_kind"] != DECLINE
    return not agent_error and not rerank_error and "tool_bug" not in record.tool_calls["rejected"]


class DryRunModel:
    """Free stand-in. As the agent: one search, then finish with every handle seen
    in reverse. As the reranker (no tools): the shown numbers reversed."""

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelResponse:
        usage = Usage(12_000, 400)
        if not tools:
            k = len(re.findall(r"^\[(\d+)\]", messages[-1].content, re.M))
            return ModelResponse(text="RANKING: " + ", ".join(str(i) for i in range(k, 0, -1)), usage=usage,
                                 stop_reason="end_turn", model=MODEL)
        if not any(m.role == "tool" for m in messages):
            claim = messages[1].content.splitlines()[0].removeprefix("Claim: ")
            return ModelResponse(tool_calls=(ToolCall("s1", "search", {"query": " ".join(claim.split()[-3:])}),),
                                 usage=usage, stop_reason="tool_use", model=MODEL)
        handles = sorted({int(n) for m in messages for n in re.findall(r"^\[D(\d+)\]", m.content, re.M)})
        answer = "RANKING: " + ", ".join(f"D{n}" for n in reversed(handles))
        return ModelResponse(tool_calls=(ToolCall("f1", "finish", {"answer": answer}),), usage=usage,
                             stop_reason="tool_use", model=MODEL)


def run_pilot(model: ModelClient, *, data_dir: Path, out: Path, paid: bool, model_settings: dict,
              within_n: int = WITHIN_N, smoke_n: int = SMOKE_N, budget_usd: float = BUDGET_USD,
              strata_file: Path = STRATA_FILE, pilot_manifest: Path = PILOT_MANIFEST,
              log: Callable[[str], None] = print) -> dict:
    check_pins(pilot_manifest)
    checks = study.preflight(REPO, out, data_dir, SCIFACT_SHA256, paid=paid, guarded=GUARDED)
    split = load_scifact(data_dir, "train")  # never the held-out split
    index = BM25Index(split.corpus, BM25Config())
    tune = dev_partition(sorted(split.queries))["tune"]
    beyond = strata(split, index, tune, CONFIG.first_k)
    strata_sha = check_strata_file(beyond, strata_file) if paid else None
    counts = {"beyond": sum(beyond.values()), "within_available": len(beyond) - sum(beyond.values())}
    if paid and counts != EXPECTED:
        raise SystemExit(f"refusing: expected strata {EXPECTED}, found {counts}")
    sample = select_sample(beyond, within_n, smoke_n)
    if paid and (len(sample["main"]) != EXPECTED["beyond"] + within_n or len(sample["smoke"]) != smoke_n):
        raise SystemExit("refusing: the sample does not have the contracted size")
    graph = agent_mod.build_agent_graph(CONFIG, model, index, split.corpus)

    (out / "private").mkdir(parents=True, exist_ok=True)
    (out / "private" / "sample.json").write_text(json.dumps(
        {**sample, "beyond": sorted(q for q in sample["main"] if beyond[q])}, indent=2) + "\n")
    path = out / "private" / "episodes.jsonl"
    guard = study.BudgetGuard(budget_usd, EPISODE_WORST_USD)
    manifest = {
        "study": "S3 pilot: search agent (S3 + forced rerank) on a stratified tune sample",
        "mode": "paid" if paid else "dry-run (scripted model and usage; no API call)",
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL, "model_settings": model_settings, "prices_as_of": PRICES_AS_OF,
        "agent_config": asdict(CONFIG), "agent_config_sha256": agent_mod.config_sha256(CONFIG),
        "agent_prompt_sha256": agent_mod.prompt_sha256(), "rerank_prompt_sha256": rerank_mod.prompt_sha256(),
        "agent_prompts": {"system": agent_mod.AGENT_SYSTEM, "instruction": agent_mod.AGENT_INSTRUCTION,
                          "search_tool_description": agent_mod.SEARCH_TOOL_DESCRIPTION},
        **bm25_provenance(BM25Config()),
        "versions": {pkg: _version(pkg) for pkg in ("anthropic", "langgraph", "bm25s", "numpy")},
        "python": sys.version.split()[0], "dataset_sha256": SCIFACT_SHA256, "strata_file_sha256": strata_sha,
        "split": "train (tune half)", "dev_partition_seed": DEV_PARTITION_SEED, "order_seed": ORDER_SEED,
        "strata_tune": counts, "within_n": within_n, "smoke_n": smoke_n,
        "scheduled_episodes": {"smoke": len(sample["smoke"]), "main": len(sample["main"])},
        "sample_sha256": study.sha256_text("\n".join(sample["main"] + ["--"] + sample["smoke"])),
        "budget_usd": budget_usd, "worst_case_call_usd": WORST_CASE_CALL_USD,
        "episode_worst_usd": EPISODE_WORST_USD, **checks,
        "script_sha256": {n: study.sha256_file(HERE / n) for n in ("run_pilot.py", "analyze.py") if (HERE / n).is_file()},
        "completed_episodes": {"smoke": 0, "main": 0}, "estimated_spend_usd": 0.0, "smoke_passed": None,
        "stopped": False, "stop_reason": None, "harness_error": None, "calls_above_worst_case": 0,
    }
    study.write_manifest(out, manifest)

    def run_phase(phase: str, queries: list[str]) -> bool:
        for i, qid in enumerate(queries, 1):
            if not guard.can_call():
                return _stop(manifest, guard.stop_reason() or "budget", log)
            started = time.perf_counter()
            try:
                record = agent_mod.run_episode(graph, qid, split.queries[qid])
            except Exception as exc:  # any graph exception is a harness error: the run is invalid
                manifest["harness_error"] = f"{type(exc).__name__}: {exc}"
                return _stop(manifest, f"harness error in a {phase} episode", log)
            wall = time.perf_counter() - started
            costs = agent_mod.call_costs(record)
            every = costs["agent"] + costs["rerank"]
            known = sum(c for c in every if c is not None)
            guard.add(known + WORST_CASE_CALL_USD * sum(c is None for c in every))
            above = sum(c is not None and c > WORST_CASE_CALL_USD for c in every)
            guard.calls_above_worst_case += above
            ok = episode_ok(record)
            study.append_record(path, {"index": i, "phase": phase, "stratum": "beyond" if beyond[qid] else "within",
                                       "wall_s": wall, "call_usd": costs, "smoke_ok": ok, **asdict(record)})
            manifest["completed_episodes"][phase] += 1
            manifest["estimated_spend_usd"] = round(guard.spent, 6)
            manifest["calls_above_worst_case"] = guard.calls_above_worst_case
            study.write_manifest(out, manifest)
            log(f"[{phase} {i:3}/{len(queries)}] stop={record.stop_reason:12} searches="
                f"{record.tool_calls['executed_searches']} ok={ok} | spent ${guard.spent:.4f}")
            if above:
                return _stop(manifest, "a call cost more than the assumed worst case: the bound is unsafe", log)
            if phase == "smoke" and not ok:
                return _stop(manifest, "smoke episode not clean (non-decline model error or tool bug)", log)
        return True

    manifest["smoke_passed"] = run_phase("smoke", sample["smoke"])
    study.write_manifest(out, manifest)
    if manifest["smoke_passed"]:  # the main phase never starts after an unclean smoke phase
        run_phase("main", sample["main"])
    manifest["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    study.write_manifest(out, manifest)
    return manifest


def _stop(manifest: dict, reason: str, log: Callable[[str], None]) -> bool:
    manifest["stopped"], manifest["stop_reason"] = True, reason
    log(f"STOP: {reason}")
    return False


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="scripted model, no API calls, temp output")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results", type=Path, default=None, help="output folder (dry run: a temp folder)")
    parser.add_argument("--env-file", type=Path, default=REPO / ".env")
    args = parser.parse_args(argv)

    if args.dry_run:
        out = args.results or Path(tempfile.mkdtemp(prefix="s3-pilot-dryrun-"))
        model, settings = DryRunModel(), {"model": "dry-run"}
    else:
        from oarp.providers.anthropic_model import AnthropicModel, AnthropicSettings

        out = args.results or RESULTS
        key = read_env_value(args.env_file, "ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit(f"ANTHROPIC_API_KEY not found in {args.env_file}")
        base = AnthropicSettings()
        model, settings = AnthropicModel(base, api_key=key), asdict(base)
    manifest = run_pilot(model, data_dir=args.data_dir, out=out, paid=not args.dry_run, model_settings=settings)
    print(f"results: {out}")
    return manifest


if __name__ == "__main__":
    main()
