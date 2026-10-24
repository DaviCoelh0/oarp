"""S2 threshold study, paid part: S1-plain (exactly as in the S1 pilot) on the
305 tune queries outside the pilot's sample, so that all 405 tune queries have
S1 rankings for choosing the S2 threshold (analyze.py).

    venv/bin/python experiments/retrieval_s2_tau/run_tune.py --dry-run   # free: scripted model, temp folder
    venv/bin/python experiments/retrieval_s2_tau/run_tune.py             # PAID

Implements the approved contract (305 calls, cap $0.70). Refuses unless the
rerank prompts and the S1-plain RerankConfig equal the pilot manifest's, and
(paid) unless the pilot's reused files match their pinned SHA-256, the code is
committed, nothing is untracked under src/, tests/ or this folder, results/
is empty and the dataset matches its pinned SHA-256. Only the train split is
loaded, and only tune-half queries are called.

Outputs: results/manifest.json (public; no query ids) and
results/private/calls.jsonl (one record per call; gitignored).
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
PILOT_RESULTS = REPO / "experiments" / "retrieval_s1_pilot" / "results"
PILOT_FILES = ("manifest.json", "private/sample.json", "private/calls.jsonl")
# The S1 pilot's files as committed in c7fc2b5 (private ones are local only).
PILOT_SHA256 = {
    "manifest.json": "a8013b6d368c50ad98a96b2ba22d792431e1dc3ec19e78658329e23e7ec45cec",
    "private/sample.json": "771a2020c58ef37aa59b258ca351eb7c7a0f382b71676c8be86f8e9a12b48021",
    "private/calls.jsonl": "4a143981f208ea8fefc5fa31ad53d6fa370f16c468518b7c63bdca1714acc50e",
}
MODEL = "claude-haiku-5-5"
CONFIG = rerank_mod.RerankConfig(k=20, shuffle=False)  # S1-plain, as in the pilot
ORDER_SEED = "oarp-r2-s2-tau-v1:"
EXPECTED_QUERIES = 305  # 405 tune queries minus the pilot's 100 (checked in paid mode)
BUDGET_USD = 0.70
WORST_CASE_CALL_USD = 0.0040  # as in the check study: largest prompt at 2.6 chars/token + 4,096 output
GUARDED = ("src/", "tests/", "experiments/retrieval_s2_tau/")


def load_pilot(pilot: Path, *, paid: bool) -> dict:
    """The pilot's manifest and sample; refuses on a config mismatch or (paid) changed files."""
    missing = [f for f in PILOT_FILES if not (pilot / f).is_file()]
    if missing:
        raise SystemExit(f"refusing: S1 pilot files missing: {missing}")
    hashes = {f: study.sha256_file(pilot / f) for f in PILOT_FILES}
    if paid:
        changed = [f for f in PILOT_FILES if hashes[f] != PILOT_SHA256[f]]
        if changed:
            raise SystemExit(f"refusing: S1 pilot files differ from the pinned SHA-256: {changed}")
    manifest = json.loads((pilot / "manifest.json").read_text())
    if manifest.get("prompt_sha256") != rerank_mod.prompt_sha256():
        raise SystemExit("refusing: the rerank prompts differ from the S1 pilot's")
    if manifest.get("conditions", {}).get("plain", {}).get("rerank_config") != asdict(CONFIG):
        raise SystemExit("refusing: the S1-plain RerankConfig differs from the S1 pilot's")
    sample = json.loads((pilot / "private" / "sample.json").read_text())["sample"]
    return {"sample": sample, "sha256": hashes, "commit": (manifest.get("git") or {}).get("commit")}


def select_queries(tune: Sequence[str], pilot_sample: Sequence[str]) -> list[str]:
    """Tune queries outside the pilot sample, in SHA-256 order of ORDER_SEED + id (label-free)."""
    if set(pilot_sample) - set(tune):
        raise SystemExit("refusing: the pilot sample is not inside the tune half")
    return study.hash_order(ORDER_SEED, sorted(set(tune) - set(pilot_sample)))


class DryRunModel:
    """Free stand-in: answers with the shown numbers reversed, and scripted usage."""

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelResponse:
        k = len(re.findall(r"^\[(\d+)\]", messages[-1].content, re.M))
        return ModelResponse(text="RANKING: " + ", ".join(str(i) for i in range(k, 0, -1)),
                             usage=Usage(1000, 50), stop_reason="end_turn", model="dry-run")


def run_tune(model: ModelClient, *, data_dir: Path, out: Path, paid: bool, model_settings: dict,
             pilot_results: Path = PILOT_RESULTS, budget_usd: float = BUDGET_USD,
             worst_case_usd: float = WORST_CASE_CALL_USD, log: Callable[[str], None] = print) -> dict:
    pilot = load_pilot(pilot_results, paid=paid)
    checks = study.preflight(REPO, out, data_dir, SCIFACT_SHA256, paid=paid, guarded=GUARDED)
    split = load_scifact(data_dir, "train")  # never the held-out split
    queries = select_queries(dev_partition(sorted(split.queries))["tune"], pilot["sample"])
    if paid and len(queries) != EXPECTED_QUERIES:
        raise SystemExit(f"refusing: expected {EXPECTED_QUERIES} tune queries outside the pilot, found {len(queries)}")
    index = BM25Index(split.corpus, BM25Config())
    bm25 = {q: [d for d, _ in index.rank(split.queries[q])] for q in queries}

    (out / "private").mkdir(parents=True, exist_ok=True)
    (out / "private" / "sample.json").write_text(json.dumps({"queries": queries}, indent=2) + "\n")
    calls_path = out / "private" / "calls.jsonl"
    guard = study.BudgetGuard(budget_usd, worst_case_usd)
    manifest = {
        "study": "S2 threshold study: S1-plain on the tune queries outside the S1 pilot sample",
        "mode": "paid" if paid else "dry-run (scripted model and usage; no API call)",
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL, "model_settings": model_settings, "prices_as_of": PRICES_AS_OF,
        "rerank": rerank_mod.provenance(CONFIG), "prompt_sha256": rerank_mod.prompt_sha256(),
        "s1_pilot": {"commit": pilot["commit"], "sha256": pilot["sha256"]},
        **bm25_provenance(BM25Config()),
        "versions": {pkg: _version(pkg) for pkg in ("anthropic", "bm25s", "numpy")},
        "python": sys.version.split()[0], "dataset_sha256": SCIFACT_SHA256,
        "split": "train (tune half minus the S1 pilot sample)", "dev_partition_seed": DEV_PARTITION_SEED,
        "order_seed": ORDER_SEED, "queries": len(queries),
        "queries_sha256": study.sha256_text("\n".join(queries)), "scheduled_calls": len(queries),
        "budget_usd": budget_usd, "worst_case_call_usd": worst_case_usd, **checks,
        "script_sha256": {n: study.sha256_file(HERE / n) for n in ("run_tune.py", "analyze.py") if (HERE / n).is_file()},
        "completed_calls": 0, "estimated_spend_usd": 0.0, "stopped_by_budget": False, "stop_reason": None,
        "calls_above_worst_case": 0,
    }
    study.write_manifest(out, manifest)
    for i, qid in enumerate(queries, 1):
        if not guard.can_call():
            manifest["stopped_by_budget"], manifest["stop_reason"] = True, guard.stop_reason()
            log(f"STOP: {guard.stop_reason()}")
            break
        record = rerank_mod.rerank(model, qid, split.queries[qid], bm25[qid], split.corpus, CONFIG)
        cost = estimate_cost(MODEL, record.usage)
        guard.add(cost)
        study.append_record(calls_path, {
            "index": i, "pass": 1, "query_id": qid, "condition": "plain",
            "bm25_top": bm25[qid][:CONFIG.k], "ranking_top": record.ranking[:CONFIG.k],
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
        log(f"[{i:3}/{len(queries)}] status={record.status:11} cost=${cost} | spent ${guard.spent:.4f}")
    if guard.calls_above_worst_case and not manifest["stopped_by_budget"]:
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
    parser.add_argument("--pilot-results", type=Path, default=PILOT_RESULTS)
    parser.add_argument("--env-file", type=Path, default=REPO / ".env")
    args = parser.parse_args(argv)

    if args.dry_run:
        out = args.results or Path(tempfile.mkdtemp(prefix="s2-tau-dryrun-"))
        model, settings = DryRunModel(), {"model": "dry-run"}
    else:
        from oarp.providers.anthropic_model import AnthropicModel, AnthropicSettings

        out = args.results or RESULTS
        key = read_env_value(args.env_file, "ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit(f"ANTHROPIC_API_KEY not found in {args.env_file}")
        base = AnthropicSettings()
        model, settings = AnthropicModel(base, api_key=key), asdict(base)
    manifest = run_tune(model, data_dir=args.data_dir, out=out, paid=not args.dry_run, model_settings=settings,
                        pilot_results=args.pilot_results)
    print(f"results: {out}")
    return manifest


if __name__ == "__main__":
    main()
