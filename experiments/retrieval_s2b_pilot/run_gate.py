"""S2b gate pilot: one cheap model call per query ("is result 1 relevant?") at
effort medium (primary) and low (secondary), on the S1 pilot's 100 tune
queries, 3 passes each: 600 calls. No rerank calls: gated rankings are built
offline from the S1 pilot's recorded rankings (analyze.py).

    venv/bin/python experiments/retrieval_s2b_pilot/run_gate.py --dry-run   # free: scripted model, temp folder
    venv/bin/python experiments/retrieval_s2b_pilot/run_gate.py             # PAID

Implements the approved contract (cap $0.50). Reads the S1 pilot's results
read-only. Paid mode refuses unless the pilot run was paid and complete and
its manifest, sample and call records match the SHA-256 values pinned below;
unless the code is committed, nothing is untracked under src/, tests/ or this
folder, results/ is empty and the dataset matches its pinned SHA-256. Every
query's top 3 is checked against a fresh BM25 ranking. The test split is
never loaded.

Outputs: results/manifest.json (public; no query ids) and
results/private/calls.jsonl (one record per call; gitignored).
"""

import argparse
import json
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from oarp.core.env import read_env_value  # noqa: E402
from oarp.core.model import ModelClient  # noqa: E402
from oarp.core.pricing import PRICES_AS_OF, estimate_cost  # noqa: E402
from oarp.core.types import Message, ModelResponse, ToolSpec, Usage  # noqa: E402
from oarp.envs.retrieval import gate as gate_mod  # noqa: E402
from oarp.envs.retrieval import study  # noqa: E402
from oarp.envs.retrieval.bm25 import BM25Config, BM25Index  # noqa: E402
from oarp.envs.retrieval.data import DEFAULT_DATA_DIR, SCIFACT_SHA256, load_scifact  # noqa: E402

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
EFFORTS = ("medium", "low")  # primary, secondary
ORDER_SEED = "oarp-r2-s2b-pilot-v1-order:"
PASSES = 3
BUDGET_USD = 0.50
# Largest gate prompt (~6,500 characters) at 2.6 characters per token (~2.5k
# input tokens) plus max_tokens = 4,096 output is ~$0.0023; rounded up.
WORST_CASE_CALL_USD = 0.0025
GUARDED = ("src/", "tests/", "experiments/retrieval_s2b_pilot/")


def load_pilot(pilot: Path, *, paid: bool) -> dict:
    """The pilot's sample and pass-1 BM25 top lists; refuses on anything unexpected."""
    missing = [f for f in PILOT_FILES if not (pilot / f).is_file()]
    if missing:
        raise SystemExit(f"refusing: S1 pilot files missing: {missing}")
    hashes = {f: study.sha256_file(pilot / f) for f in PILOT_FILES}
    manifest = json.loads((pilot / "manifest.json").read_text())
    if paid:
        if manifest.get("mode") != "paid" or manifest.get("completed_calls") != manifest.get("scheduled_calls"):
            raise SystemExit("refusing: the S1 pilot run is not a complete paid run")
        changed = [f for f in PILOT_FILES if hashes[f] != PILOT_SHA256[f]]
        if changed:
            raise SystemExit(f"refusing: S1 pilot files differ from the pinned SHA-256: {changed}")
    sample = json.loads((pilot / "private" / "sample.json").read_text())["sample"]
    rows = [json.loads(line) for line in (pilot / "private" / "calls.jsonl").read_text().splitlines() if line]
    top = {r["query_id"]: r["bm25_top"] for r in rows if r["pass"] == 1 and r["condition"] == "plain"}
    if set(sample) - set(top):
        raise SystemExit("refusing: the S1 pilot has no pass-1 plain record for some sample queries")
    return {"sample": sample, "bm25_top": top, "sha256": hashes, "commit": (manifest.get("git") or {}).get("commit")}


def build_schedule(sample: Sequence[str], passes: int = PASSES) -> list[dict]:
    schedule = []
    for pass_no in range(1, passes + 1):
        for qid in sample:
            for effort in study.first_of_pair(ORDER_SEED, qid, EFFORTS):
                schedule.append({"index": len(schedule) + 1, "pass": pass_no, "query_id": qid, "effort": effort})
    return schedule


class DryRunModel:
    """Free stand-in: says "yes" when the prompt length is even, with scripted usage."""

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelResponse:
        answer = "yes" if len(messages[-1].content) % 2 == 0 else "no"
        return ModelResponse(text=f"ANSWER: {answer}", usage=Usage(1700, 20), stop_reason="end_turn", model="dry-run")


def run_gate(models: dict[str, ModelClient], *, data_dir: Path, out: Path, paid: bool, model_settings: dict,
             pilot_results: Path = PILOT_RESULTS, budget_usd: float = BUDGET_USD,
             worst_case_usd: float = WORST_CASE_CALL_USD, log: Callable[[str], None] = print) -> dict:
    pilot = load_pilot(pilot_results, paid=paid)
    checks = study.preflight(REPO, out, data_dir, SCIFACT_SHA256, paid=paid, guarded=GUARDED)
    split = load_scifact(data_dir, "train")  # never the held-out split
    index = BM25Index(split.corpus, BM25Config())
    for qid in pilot["sample"]:
        fresh = [d for d, _ in index.rank(split.queries[qid], gate_mod.GATE_TOP)]
        if fresh != pilot["bm25_top"][qid][:gate_mod.GATE_TOP]:
            raise SystemExit("refusing: a recorded BM25 top 3 differs from a fresh BM25 ranking")
    schedule = build_schedule(pilot["sample"])

    (out / "private").mkdir(parents=True, exist_ok=True)
    calls_path = out / "private" / "calls.jsonl"
    guard = study.BudgetGuard(budget_usd, worst_case_usd)
    manifest = {
        "study": "S2b gate pilot: model gate at effort medium (primary) and low, on the S1 pilot's queries",
        "mode": "paid" if paid else "dry-run (scripted model and usage; no API call)",
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL, "model_settings": model_settings, "efforts": list(EFFORTS), "prices_as_of": PRICES_AS_OF,
        "gate": gate_mod.provenance(),
        "prompt_sha256": {"gate_system": study.sha256_text(gate_mod.GATE_SYSTEM),
                          "gate_instruction": study.sha256_text(gate_mod.GATE_INSTRUCTION)},
        "s1_pilot": {"commit": pilot["commit"], "sha256": pilot["sha256"]},
        "versions": {pkg: _version(pkg) for pkg in ("anthropic", "bm25s", "numpy")},
        "python": sys.version.split()[0], "dataset_sha256": SCIFACT_SHA256, "split": "train (tune half, S1 pilot sample)",
        "order_seed": ORDER_SEED, "queries": len(pilot["sample"]), "passes": PASSES,
        "queries_sha256": study.sha256_text("\n".join(pilot["sample"])), "scheduled_calls": len(schedule),
        "budget_usd": budget_usd, "worst_case_call_usd": worst_case_usd, **checks,
        "script_sha256": {n: study.sha256_file(HERE / n) for n in ("run_gate.py", "analyze.py") if (HERE / n).is_file()},
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
        record = gate_mod.gate(models[entry["effort"]], split.queries[qid], pilot["bm25_top"][qid], split.corpus)
        cost = estimate_cost(MODEL, record.usage)
        guard.add(cost)
        study.append_record(calls_path, {
            **entry, "status": record.status, "rerank": not record.keep_bm25,
            "usage": asdict(record.usage) if record.usage else None, "estimated_usd": cost,
            "latency_s": record.latency_s, "model_reported": record.model, "stop_reason": record.stop_reason,
            "error_kind": record.error_kind, "error": record.error, "text": record.text,
            "reasoning_summary": record.reasoning_summary})
        manifest["completed_calls"] += 1
        manifest["estimated_spend_usd"] = round(guard.spent, 6)
        manifest["calls_above_worst_case"] = guard.calls_above_worst_case
        study.write_manifest(out, manifest)
        log(f"[{entry['index']:3}/{len(schedule)}] pass {entry['pass']} {entry['effort']:6} "
            f"status={record.status:11} cost=${cost} | spent ${guard.spent:.4f}")
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
        out = args.results or Path(tempfile.mkdtemp(prefix="s2b-dryrun-"))
        models = {e: DryRunModel() for e in EFFORTS}
        settings = {e: {"model": "dry-run", "effort": e} for e in EFFORTS}
    else:
        from oarp.providers.anthropic_model import AnthropicModel, AnthropicSettings

        out = args.results or RESULTS
        key = read_env_value(args.env_file, "ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit(f"ANTHROPIC_API_KEY not found in {args.env_file}")
        base = AnthropicSettings()
        models = {e: AnthropicModel(replace(base, effort=e), api_key=key) for e in EFFORTS}
        settings = {e: asdict(replace(base, effort=e)) for e in EFFORTS}
    manifest = run_gate(models, data_dir=args.data_dir, out=out, paid=not args.dry_run, model_settings=settings,
                        pilot_results=args.pilot_results)
    print(f"results: {out}")
    return manifest


if __name__ == "__main__":
    main()
