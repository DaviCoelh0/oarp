"""S1 pilot: listwise reranking of BM25's top 20 by Claude Haiku 5.5, shown in
BM25 order (S1-plain, primary) or in a seeded per-query shuffle (S1-shuffled).

    venv/bin/python experiments/retrieval_s1_pilot/run_pilot.py --dry-run   # free: scripted model, temp folder
    venv/bin/python experiments/retrieval_s1_pilot/run_pilot.py             # PAID

Implements the approved study contract: 100 tune queries (dev partition), both
conditions per query back to back (which goes first: a hash bit of the query
id), then passes 2 and 3 on the first 20 for stability: 280 calls in total.
The test split is never loaded.

Paid mode refuses to start unless tracked files are committed, nothing is
untracked under src/, tests/ or this folder, results/ is empty, the dataset
matches its pinned SHA-256 and the API key is in .env (the key goes only to
the adapter; it is never printed or written). A budget guard stops before any
call whose worst case could push the ESTIMATED spend over the cap; a call
without usage counts at the worst case. The Console spend limit is the hard
ceiling.

Outputs: results/manifest.json (public: settings, hashes, counts, spend; no
query ids) and results/private/calls.jsonl (one record per call, appended
after every call; gitignored). Score them with analyze.py (free).
"""

import argparse
import hashlib
import json
import re
import subprocess
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
from oarp.core.trace import git_info  # noqa: E402
from oarp.core.types import Message, ModelResponse, ToolSpec, Usage  # noqa: E402
from oarp.envs.retrieval import rerank as rerank_mod  # noqa: E402
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
SCRIPTS = ("run_pilot.py", "analyze.py")
MODEL = "claude-haiku-5-5"  # priced by this id; the id the API reports is recorded per call
CONDITIONS = {"plain": rerank_mod.RerankConfig(k=20, shuffle=False),
              "shuffled": rerank_mod.RerankConfig(k=20, shuffle=True)}
SAMPLE_SEED = "oarp-r2-s1-pilot-v1:"
ORDER_SEED = "oarp-r2-s1-pilot-v1-order:"
SAMPLE_SIZE = 100
STABILITY_SIZE = 20
PASSES = 3
BUDGET_USD = 0.80
# Largest possible prompt: 20 x 2,000 characters plus overhead, at a conservative
# 3 characters per token (~13.7k input tokens), plus max_tokens = 4,096 output.
WORST_CASE_CALL_USD = 0.0035
UNTRACKED_GUARDED = ("src/", "tests/", "experiments/retrieval_s1_pilot/")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def select_sample(tune_ids: Sequence[str], size: int = SAMPLE_SIZE,
                  stability: int = STABILITY_SIZE) -> tuple[list[str], list[str]]:
    """The first `size` tune ids in SHA-256 order of SAMPLE_SEED + id; the first
    `stability` of those form the stability subset. Uses no labels."""
    ordered = sorted(set(tune_ids), key=lambda q: _sha(SAMPLE_SEED + q))
    sample = ordered[:size]
    return sample, sample[:stability]


def first_condition(query_id: str) -> str:
    """Which condition runs first for this query: the first bit of SHA-256(ORDER_SEED + id)."""
    return "plain" if hashlib.sha256((ORDER_SEED + query_id).encode()).digest()[0] < 128 else "shuffled"


def build_schedule(sample: Sequence[str], subset: Sequence[str], passes: int = PASSES) -> list[dict]:
    schedule = []
    for pass_no in range(1, passes + 1):
        for qid in sample if pass_no == 1 else subset:
            first = first_condition(qid)
            for condition in (first, "shuffled" if first == "plain" else "plain"):
                schedule.append({"index": len(schedule) + 1, "pass": pass_no, "query_id": qid,
                                 "condition": condition})
    return schedule


def call_cost(usage: Usage | None) -> float | None:
    return estimate_cost(MODEL, usage)


def untracked_guarded(repo: Path) -> list[str]:
    out = subprocess.run(["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard"],
                         capture_output=True, text=True, timeout=10).stdout.splitlines()
    return [p for p in out if p.startswith(UNTRACKED_GUARDED)]


def preflight(data_dir: Path, out: Path, *, paid: bool, repo: Path = REPO) -> dict:
    git = git_info(repo)
    if paid:
        if git is None or git["dirty"]:
            raise SystemExit(f"refusing a paid run: tracked files have uncommitted changes ({git})")
        stray = untracked_guarded(repo)
        if stray:
            raise SystemExit(f"refusing a paid run: untracked files under guarded folders: {stray}")
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{out} is not empty; refusing to mix runs")
        archive = data_dir / "scifact" / "ARCHIVE.json"
        sha = json.loads(archive.read_text()).get("sha256") if archive.is_file() else None
        if sha != SCIFACT_SHA256:
            raise SystemExit(f"refusing a paid run: dataset SHA-256 {sha} is not the pinned {SCIFACT_SHA256}")
    scripts = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
               for name in SCRIPTS if (HERE / name).is_file()}
    prompts = {"rerank_system": _sha(rerank_mod.RERANK_SYSTEM),
               "rerank_instruction": _sha(rerank_mod.RERANK_INSTRUCTION)}
    return {"git": git, "script_sha256": scripts, "prompt_sha256": prompts}


class DryRunModel:
    """Free stand-in: answers with the shown numbers reversed, and scripted usage."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelResponse:
        self.calls += 1
        k = len(re.findall(r"^\[(\d+)\]", messages[-1].content, re.M))
        return ModelResponse(text="RANKING: " + ", ".join(str(i) for i in range(k, 0, -1)),
                             usage=Usage(1000, 50), stop_reason="end_turn", model="dry-run")


def run_pilot(model: ModelClient, *, data_dir: Path, out: Path, paid: bool, model_settings: dict,
              sample_size: int = SAMPLE_SIZE, stability: int = STABILITY_SIZE, budget_usd: float = BUDGET_USD,
              worst_case_usd: float = WORST_CASE_CALL_USD,
              log: Callable[[str], None] = print) -> dict:
    checks = preflight(data_dir, out, paid=paid)
    split = load_scifact(data_dir, "train")  # never the held-out split
    tune = dev_partition(sorted(split.queries))["tune"]
    sample, subset = select_sample(tune, sample_size, stability)
    schedule = build_schedule(sample, subset)
    index = BM25Index(split.corpus, BM25Config())
    bm25 = {q: [d for d, _ in index.rank(split.queries[q])] for q in sample}

    (out / "private").mkdir(parents=True, exist_ok=True)
    (out / "private" / "sample.json").write_text(json.dumps({"sample": sample, "stability": subset}, indent=2) + "\n")
    calls_path = out / "private" / "calls.jsonl"
    manifest = {
        "study": "S1 pilot: listwise reranking of BM25 top 20, plain vs shuffled presentation",
        "mode": "paid" if paid else "dry-run (scripted model and usage; no API call)",
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL, "model_settings": model_settings, "prices_as_of": PRICES_AS_OF,
        "conditions": {name: rerank_mod.provenance(config) for name, config in CONDITIONS.items()},
        **bm25_provenance(BM25Config()),
        "versions": {pkg: _version(pkg) for pkg in ("anthropic", "bm25s", "numpy")},
        "python": sys.version.split()[0], "dataset_sha256": SCIFACT_SHA256, "split": "train (tune half)",
        "dev_partition_seed": DEV_PARTITION_SEED, "sample_seed": SAMPLE_SEED, "order_seed": ORDER_SEED,
        "sample_size": len(sample), "stability_size": len(subset), "passes": PASSES,
        "sample_sha256": _sha("\n".join(sample)), "scheduled_calls": len(schedule),
        "budget_usd": budget_usd, "worst_case_call_usd": worst_case_usd, **checks,
        "completed_calls": 0, "estimated_spend_usd": 0.0, "stopped_by_budget": False,
        "calls_above_worst_case": 0,  # > 0 means the guard's bound was wrong: the run stops
    }
    write_manifest(out, manifest)
    spent = 0.0
    for entry in schedule:
        if spent + worst_case_usd > budget_usd:
            log(f"STOP: estimated spend ${spent:.4f} + worst case ${worst_case_usd} would exceed ${budget_usd}")
            manifest["stopped_by_budget"] = True
            break
        qid, config = entry["query_id"], CONDITIONS[entry["condition"]]
        record = rerank_mod.rerank(model, qid, split.queries[qid], bm25[qid], split.corpus, config)
        cost = call_cost(record.usage)
        spent += worst_case_usd if cost is None else cost
        if cost is not None and cost > worst_case_usd:
            manifest["calls_above_worst_case"] += 1
        k = len(bm25[qid][:config.k])  # fewer than k only on tiny fixture corpora
        shown = rerank_mod.presentation_order(qid, k) if config.shuffle else list(range(k))
        row = {**entry, "bm25_top": bm25[qid][:config.k], "shown_bm25_positions": shown,
               "ranking_top": record.ranking[:config.k], "status": record.status,
               "duplicates": record.duplicates, "out_of_range": record.out_of_range, "missing": record.missing,
               "copied_presentation_order": record.copied_presentation_order,
               "matches_bm25_order": record.matches_bm25_order,
               "usage": asdict(record.usage) if record.usage else None, "estimated_usd": cost,
               "latency_s": record.latency_s, "model_reported": record.model, "stop_reason": record.stop_reason,
               "error_kind": record.error_kind, "error": record.error, "text": record.text,
               "reasoning_summary": record.reasoning_summary}
        with calls_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        manifest["completed_calls"] += 1
        manifest["estimated_spend_usd"] = round(spent, 6)
        write_manifest(out, manifest)  # after every call: a crash still leaves an accurate record
        log(f"[{entry['index']:3}/{len(schedule)}] pass {entry['pass']} {entry['condition']:8} "
            f"status={record.status:11} cost=${cost} | spent ${spent:.4f}")
        if manifest["calls_above_worst_case"]:
            log(f"STOP: a call cost ${cost} > the assumed worst case ${worst_case_usd}; the budget guard is unsafe")
            break
    manifest["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_manifest(out, manifest)
    return manifest


def write_manifest(out: Path, manifest: dict) -> None:
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


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
        out = args.results or Path(tempfile.mkdtemp(prefix="s1-pilot-dryrun-"))
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
