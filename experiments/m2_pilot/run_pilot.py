"""M2 pilot: feasibility of the first two M2 dev tasks (2 tasks x 3 repetitions).

    venv/bin/python experiments/m2_pilot/run_pilot.py --dry-run   # free: checks + scripted model
    venv/bin/python experiments/m2_pilot/run_pilot.py             # PAID: Claude Haiku 5.5

A feasibility check, not a comparison: one configuration (P4's frozen
settings, run_tests on). Before any run it checks, for free, that tracked
files are committed (paid mode), that every task passes the static and
verifier-judged validity checks, and that the task set matches a manifest
frozen at the start. Runs are interleaved round-robin over tasks. A budget
guard stops before any run whose worst case could push the ESTIMATED spend
over the cap; the Console spend limit is the hard ceiling.
"""

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from oarp.core.env import read_env_value  # noqa: E402
from oarp.core.model import ScriptedModel  # noqa: E402
from oarp.core.trace import git_info  # noqa: E402
from oarp.core.types import ModelResponse, ToolCall  # noqa: E402
from oarp.envs.coding.manifest import check_manifest, write_manifest  # noqa: E402
from oarp.envs.coding.runner import HarnessConfig, run_coding_task  # noqa: E402
from oarp.envs.coding.task import CodingTask, discover_tasks  # noqa: E402
from oarp.envs.coding.validity import behaviour_problems, static_problems  # noqa: E402

TASKS_DIR = REPO / "tasks"
TASK_IDS = ("feed_pagination_006", "config_merge_007")
RESULTS = Path(__file__).resolve().parent / "results"
REPETITIONS = 3
HARNESS = HarnessConfig(max_steps=12, max_tool_calls_per_step=1, run_tests_enabled=True, loop="plain")
BUDGET_USD = 0.40
WORST_CASE_RUN_USD = 0.05  # 12 calls x (15K input + 4,096 output tokens) on Haiku 5.5, prompt <= 100K
SCRIPTS = ("run_pilot.py", "analyze.py")


def select_tasks() -> list[CodingTask]:
    available = {t.id: t for t in discover_tasks(TASKS_DIR)}  # held-out tasks are never loaded
    missing = [i for i in TASK_IDS if i not in available]
    if missing:
        raise SystemExit(f"tasks not found: {missing}")
    return [available[i] for i in TASK_IDS]


def preflight(tasks: list[CodingTask], out: Path, *, paid: bool) -> dict:
    git = git_info(REPO)
    if paid and (git is None or git["dirty"]):
        raise SystemExit(f"refusing a paid run: tracked files have uncommitted changes ({git})")
    for task in tasks:
        problems = static_problems(task) + behaviour_problems(task)
        if problems:
            raise SystemExit(f"task {task.id} failed validation: {problems}")
        print(f"validated {task.id}")
    manifest_path = out / "task_manifest.json"
    write_manifest(tasks, manifest_path)
    assert check_manifest(manifest_path, TASKS_DIR) == []
    scripts = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in SCRIPTS}
    return {"git": git, "script_sha256": scripts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="scripted model, no API calls, temp output")
    args = parser.parse_args()

    tasks = select_tasks()
    schedule = [task for _ in range(REPETITIONS) for task in tasks]  # round-robin

    if args.dry_run:
        out = Path(tempfile.mkdtemp(prefix="m2-pilot-dryrun-"))
        settings = {"model": "scripted-dry-run"}
        make_model = lambda: ScriptedModel([  # noqa: E731
            ModelResponse(tool_calls=(ToolCall("c1", "run_tests", {}),)),
            ModelResponse(tool_calls=(ToolCall("c2", "finish", {"answer": "dry run"}),)),
        ])
    else:
        from oarp.providers.anthropic_model import AnthropicModel, AnthropicSettings

        out = RESULTS
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{out} is not empty; refusing to mix pilot runs")
        key = read_env_value(REPO / ".env", "ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit("ANTHROPIC_API_KEY not found in .env")
        model_settings = AnthropicSettings()
        settings = asdict(model_settings)
        model = AnthropicModel(model_settings, api_key=key)
        make_model = lambda: model  # noqa: E731  (stateless: one client for all runs)

    out.mkdir(parents=True, exist_ok=True)
    checks = preflight(tasks, out, paid=not args.dry_run)
    manifest = {
        "study": "M2 pilot: feasibility of two dev tasks (not a comparison, not a benchmark)",
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "harness": asdict(HARNESS), "model_settings": settings,
        "tasks": list(TASK_IDS), "repetitions": REPETITIONS,
        "schedule": [t.id for t in schedule], "budget_usd": BUDGET_USD,
        "worst_case_run_usd": WORST_CASE_RUN_USD, **checks, "runs": [],
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
        write_run_manifest(out, manifest)  # after every run: a crash still leaves an accurate record
        print(f"[{index}/{len(schedule)}] {task.id:20} valid={summary['valid']!s:5} "
              f"success={summary.get('verified_success')!s:5} public={summary.get('public_status')} "
              f"hidden={summary.get('hidden_status')} stop={summary.get('stop_reason')} "
              f"calls={summary.get('model_calls')} cost=${cost} | spent ${spent:.4f}")

    problems = check_manifest(out / "task_manifest.json", TASKS_DIR)
    manifest["task_manifest_unchanged_at_end"] = not problems
    manifest["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["estimated_spend_usd"] = round(spent, 6)
    write_run_manifest(out, manifest)
    print(f"results: {out}" + (f"\nWARNING: tasks changed during the pilot: {problems}" if problems else ""))


def write_run_manifest(out: Path, manifest: dict) -> None:
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
