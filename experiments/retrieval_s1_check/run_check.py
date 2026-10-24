"""S1 confirmation on the dev check half: S1-plain exactly as in the S1 pilot,
on all 404 check-half queries, plus passes 2 and 3 on 20 of them (stability).

    venv/bin/python experiments/retrieval_s1_check/run_check.py --dry-run   # free: scripted model, temp folder
    venv/bin/python experiments/retrieval_s1_check/run_check.py             # PAID

Implements the approved contract (444 calls, cap $1.00). Refuses to run if the
rerank prompts or the S1-plain RerankConfig differ from the ones recorded in
the S1 pilot's manifest, so nothing can change between pilot and confirmation.
Paid mode also refuses unless the code is committed, nothing is untracked
under src/, tests/ or this folder, results/ is empty and the dataset matches
its pinned SHA-256. The test split is never loaded.

Outputs: results/manifest.json (public; no query ids) and
results/private/calls.jsonl (one record per call; gitignored). Score with
analyze.py (free).
"""

import argparse
import json
import re
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from oarp.core.env import read_env_value  # noqa: E402
from oarp.core.model import ModelClient  # noqa: E402
from oarp.core.pricing import PRICES_AS_OF, estimate_cost  # noqa: E402
from oarp.core.types import Message, ModelResponse, ToolSpec, Usage  # noqa: E402
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
PILOT_MANIFEST = REPO / "experiments" / "retrieval_s1_pilot" / "results" / "manifest.json"
MODEL = "claude-haiku-5-5"
CONFIG = rerank_mod.RerankConfig(k=20, shuffle=False)  # S1-plain, as in the pilot
STABILITY_SEED = "oarp-r2-s1-check-v1:"
STABILITY_SIZE = 20
PASSES = 3
BUDGET_USD = 1.00
# Largest prompt at the pilot's lowest observed 2.6 characters per token
# (~15.8k input tokens) plus max_tokens = 4,096 output is ~$0.0036; rounded up.
WORST_CASE_CALL_USD = 0.0040
GUARDED = ("src/", "tests/", "experiments/retrieval_s1_check/")


def prompt_hashes() -> dict[str, str]:
    return {"rerank_system": study.sha256_text(rerank_mod.RERANK_SYSTEM),
            "rerank_instruction": study.sha256_text(rerank_mod.RERANK_INSTRUCTION)}


def check_same_as_pilot(pilot_manifest: Path) -> dict:
    """Refuse unless prompts and the S1-plain config equal the pilot's recorded ones."""
    if not pilot_manifest.is_file():
        raise SystemExit(f"pilot manifest not found: {pilot_manifest}")
    pilot = json.loads(pilot_manifest.read_text())
    if pilot.get("prompt_sha256") != prompt_hashes():
        raise SystemExit("refusing: the rerank prompts differ from the S1 pilot's")
    if pilot.get("conditions", {}).get("plain", {}).get("rerank_config") != asdict(CONFIG):
        raise SystemExit("refusing: the S1-plain RerankConfig differs from the S1 pilot's")
    return {"pilot_manifest_sha256": study.sha256_file(pilot_manifest), "pilot_commit": (pilot.get("git") or {}).get("commit")}


def build_schedule(queries: Sequence[str], subset: Sequence[str], passes: int = PASSES) -> list[dict]:
    schedule = []
    for pass_no in range(1, passes + 1):
        for qid in queries if pass_no == 1 else subset:
            schedule.append({"index": len(schedule) + 1, "pass": pass_no, "query_id": qid, "condition": "plain"})
    return schedule


class DryRunModel:
    """Free stand-in: answers with the shown numbers reversed, and scripted usage."""

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelResponse:
        k = len(re.findall(r"^\[(\d+)\]", messages[-1].content, re.M))
        return ModelResponse(text="RANKING: " + ", ".join(str(i) for i in range(k, 0, -1)),
                             usage=Usage(1000, 50), stop_reason="end_turn", model="dry-run")


def run_check(model: ModelClient, *, data_dir: Path, out: Path, paid: bool, model_settings: dict,
              pilot_manifest: Path = PILOT_MANIFEST, stability: int = STABILITY_SIZE,
              budget_usd: float = BUDGET_USD, worst_case_usd: float = WORST_CASE_CALL_USD,
              log: Callable[[str], None] = print) -> dict:
    same = check_same_as_pilot(pilot_manifest)
    checks = study.preflight(REPO, out, data_dir, SCIFACT_SHA256, paid=paid, guarded=GUARDED)
    split = load_scifact(data_dir, "train")  # never the held-out split
    queries = dev_partition(sorted(split.queries))["check"]
    subset = study.hash_order(STABILITY_SEED, queries)[:stability]
    schedule = build_schedule(queries, subset)
    index = BM25Index(split.corpus, BM25Config())
    bm25 = {q: [d for d, _ in index.rank(split.queries[q])] for q in queries}

    (out / "private").mkdir(parents=True, exist_ok=True)
    (out / "private" / "sample.json").write_text(json.dumps({"queries": queries, "stability": subset}, indent=2) + "\n")
    calls_path = out / "private" / "calls.jsonl"
    guard = study.BudgetGuard(budget_usd, worst_case_usd)
    manifest = {
        "study": "S1 confirmation on the dev check half (S1-plain as in the pilot)",
        "mode": "paid" if paid else "dry-run (scripted model and usage; no API call)",
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL, "model_settings": model_settings, "prices_as_of": PRICES_AS_OF,
        "rerank": rerank_mod.provenance(CONFIG), "prompt_sha256": prompt_hashes(), **same,
        **bm25_provenance(BM25Config()),
        "versions": {pkg: _version(pkg) for pkg in ("anthropic", "bm25s", "numpy")},
        "python": sys.version.split()[0], "dataset_sha256": SCIFACT_SHA256, "split": "train (check half)",
        "dev_partition_seed": DEV_PARTITION_SEED, "stability_seed": STABILITY_SEED,
        "queries": len(queries), "stability_size": len(subset), "passes": PASSES,
        "queries_sha256": study.sha256_text("\n".join(queries)), "scheduled_calls": len(schedule),
        "budget_usd": budget_usd, "worst_case_call_usd": worst_case_usd, **checks,
        "script_sha256": {n: study.sha256_file(HERE / n) for n in ("run_check.py", "analyze.py") if (HERE / n).is_file()},
        "completed_calls": 0, "estimated_spend_usd": 0.0, "stopped_by_budget": False, "stop_reason": None,
        "calls_above_worst_case": 0,
    }
    study.write_manifest(out, manifest)
    for entry in schedule:
        if not guard.can_call():
            manifest["stopped_by_budget"], manifest["stop_reason"] = True, guard.stop_reason()
            log(f"STOP: {guard.stop_reason()}")
            break
        qid = entry["query_id"]
        record = rerank_mod.rerank(model, qid, split.queries[qid], bm25[qid], split.corpus, CONFIG)
        cost = estimate_cost(MODEL, record.usage)
        guard.add(cost)
        study.append_record(calls_path, {
            **entry, "bm25_top": bm25[qid][:CONFIG.k], "ranking_top": record.ranking[:CONFIG.k],
            "status": record.status, "duplicates": record.duplicates, "out_of_range": record.out_of_range,
            "missing": record.missing, "copied_presentation_order": record.copied_presentation_order,
            "matches_bm25_order": record.matches_bm25_order,
            "usage": asdict(record.usage) if record.usage else None, "estimated_usd": cost,
            "latency_s": record.latency_s, "model_reported": record.model, "stop_reason": record.stop_reason,
            "error_kind": record.error_kind, "error": record.error, "text": record.text,
            "reasoning_summary": record.reasoning_summary})
        manifest["completed_calls"] += 1
        manifest["estimated_spend_usd"] = round(guard.spent, 6)
        manifest["calls_above_worst_case"] = guard.calls_above_worst_case
        study.write_manifest(out, manifest)
        log(f"[{entry['index']:3}/{len(schedule)}] pass {entry['pass']} status={record.status:11} "
            f"cost=${cost} | spent ${guard.spent:.4f}")
    if not manifest["stopped_by_budget"] and not guard.can_call() and guard.calls_above_worst_case:
        manifest["stopped_by_budget"], manifest["stop_reason"] = True, guard.stop_reason()
    manifest["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    study.write_manifest(out, manifest)
    return manifest


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
        out = args.results or Path(tempfile.mkdtemp(prefix="s1-check-dryrun-"))
        model, settings = DryRunModel(), {"model": "dry-run"}
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
