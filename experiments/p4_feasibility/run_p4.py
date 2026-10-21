"""P4: preliminary coding-agent feasibility study (5 dev tasks x 3 repetitions).

    venv/bin/python experiments/p4_feasibility/run_p4.py --dry-run   # free: scripted model
    venv/bin/python experiments/p4_feasibility/run_p4.py             # PAID: Claude Haiku 5.5

Runs are interleaved round-robin over tasks so a time-dependent effect (e.g.
API latency drifting during the session) does not line up with one task.
A budget guard stops before any run whose worst case could push the
ESTIMATED spend over the cap; the Console spend limit is the hard ceiling.
"""

import argparse
import json
import re
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from oarp.core.model import ScriptedModel  # noqa: E402
from oarp.core.types import ModelResponse, ToolCall  # noqa: E402
from oarp.envs.coding.runner import HarnessConfig, run_coding_task  # noqa: E402
from oarp.envs.coding.task import discover_tasks  # noqa: E402

TASK_DIR = REPO / "tasks" / "coding" / "dev"
RESULTS = Path(__file__).resolve().parent / "results"
REPETITIONS = 3
HARNESS = HarnessConfig(max_steps=12, max_tool_calls_per_step=1, run_tests_enabled=True)
BUDGET_USD = 1.00
WORST_CASE_RUN_USD = 0.04  # 12 calls x (10K input + 4,096 output tokens) on Haiku 5.5


def api_key() -> str:
    match = re.search(r'^\s*(?:export\s+)?ANTHROPIC_API_KEY\s*=\s*["\']?([^"\'\s]+)', (REPO / ".env").read_text(), re.M)
    if not match:
        raise SystemExit("ANTHROPIC_API_KEY not found in .env")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="scripted model, no API calls, temp output")
    args = parser.parse_args()

    tasks = discover_tasks(TASK_DIR)
    assert all(t.split == "dev" for t in tasks)
    schedule = [task for _ in range(REPETITIONS) for task in tasks]  # round-robin

    if args.dry_run:
        out = Path(tempfile.mkdtemp(prefix="p4-dryrun-"))
        settings = {"model": "scripted-dry-run"}
        make_model = lambda: ScriptedModel([  # noqa: E731
            ModelResponse(tool_calls=(ToolCall("c1", "run_tests", {}),)),
            ModelResponse(tool_calls=(ToolCall("c2", "finish", {"answer": "dry run"}),)),
        ])
    else:
        from oarp.providers.anthropic_model import AnthropicModel, AnthropicSettings

        out = RESULTS
        model_settings = AnthropicSettings()
        settings = asdict(model_settings)
        model = AnthropicModel(model_settings, api_key=api_key())
        make_model = lambda: model  # noqa: E731  (stateless: one client for all runs)

    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "study": "P4 preliminary feasibility (not a benchmark)",
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "harness": asdict(HARNESS), "model_settings": settings,
        "tasks": [t.id for t in tasks], "repetitions": REPETITIONS,
        "schedule": [t.id for t in schedule], "budget_usd": BUDGET_USD, "runs": [],
    }
    spent = 0.0
    for index, task in enumerate(schedule, 1):
        if spent + WORST_CASE_RUN_USD > BUDGET_USD:
            print(f"STOP: estimated spend ${spent:.4f} + worst case ${WORST_CASE_RUN_USD} would exceed ${BUDGET_USD}")
            manifest["stopped_by_budget"] = True
            break
        summary = run_coding_task(task, make_model(), HARNESS, model_settings=settings, runs_dir=out)
        cost = summary.get("estimated_cost_usd")
        spent += WORST_CASE_RUN_USD if cost is None else cost  # unknown cost: assume the worst
        manifest["runs"].append(summary["run_id"])
        manifest["estimated_spend_usd"] = round(spent, 6)
        write_manifest(out, manifest)  # after every run: a crash still leaves an accurate record
        print(f"[{index:2}/{len(schedule)}] {task.id:22} valid={summary['valid']!s:5} "
              f"success={summary.get('verified_success')!s:5} stop={summary.get('stop_reason')} "
              f"calls={summary.get('model_calls')} cost=${cost} | spent ${spent:.4f}")

    manifest["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["estimated_spend_usd"] = round(spent, 6)
    write_manifest(out, manifest)
    print(f"results: {out}")


def write_manifest(out: Path, manifest: dict) -> None:
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
