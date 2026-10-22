"""M2 effort pilot: low vs medium effort on D1 and D2 (2 tasks x 2 efforts x 3 repetitions).

    venv/bin/python experiments/m2_effort_pilot/run_effort_pilot.py --dry-run   # free
    venv/bin/python experiments/m2_effort_pilot/run_effort_pilot.py             # PAID

A model-configuration feasibility study, not a harness comparison: the only
variable is `effort`; model, every other setting, prompts, tools and the
12-call budget are identical. The four (task, effort) conditions are
interleaved so time-dependent effects don't line up with one condition.
Before any run it checks, for free, that tracked files are committed (paid
mode), that both tasks pass the validity checks, and that the task set
matches a manifest frozen at the start. A budget guard stops before any run
whose worst case could push the ESTIMATED spend over the cap; the Console
spend limit is the hard ceiling.
"""

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import asdict, replace
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
EFFORTS = ("medium", "low")
RESULTS = Path(__file__).resolve().parent / "results"
REPETITIONS = 3
HARNESS = HarnessConfig(max_steps=12, max_tool_calls_per_step=1, run_tests_enabled=True, loop="plain")
BUDGET_USD = 0.20
WORST_CASE_RUN_USD = 0.05  # 12 calls x (15K input + 4,096 output tokens) on Haiku 5.5, prompt <= 100K
SCRIPTS = ("run_effort_pilot.py", "analyze.py")


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
    # Interleaved: every repetition visits all four (task, effort) conditions.
    schedule = [(task, effort) for _ in range(REPETITIONS) for task in tasks for effort in EFFORTS]

    if args.dry_run:
        out = Path(tempfile.mkdtemp(prefix="m2-effort-dryrun-"))
        settings_for = lambda effort: {"model": "scripted-dry-run", "effort": effort}  # noqa: E731
        make_model = lambda effort: ScriptedModel([  # noqa: E731
            ModelResponse(tool_calls=(ToolCall("c1", "run_tests", {}),)),
            ModelResponse(tool_calls=(ToolCall("c2", "finish", {"answer": "dry run"}),)),
        ])
    else:
        from oarp.providers.anthropic_model import AnthropicModel, AnthropicSettings

        out = RESULTS
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{out} is not empty; refusing to mix runs")
        key = read_env_value(REPO / ".env", "ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit("ANTHROPIC_API_KEY not found in .env")
        base = AnthropicSettings()
        models = {e: AnthropicModel(replace(base, effort=e), api_key=key) for e in EFFORTS}
        settings_for = lambda effort: asdict(replace(base, effort=effort))  # noqa: E731
        make_model = lambda effort: models[effort]  # noqa: E731  (stateless clients)

    out.mkdir(parents=True, exist_ok=True)
    checks = preflight(tasks, out, paid=not args.dry_run)
    manifest = {
        "study": "M2 effort pilot: low vs medium effort on two dev tasks (model-configuration feasibility)",
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "harness": asdict(HARNESS), "efforts": list(EFFORTS),
        "model_settings": {e: settings_for(e) for e in EFFORTS},
        "tasks": list(TASK_IDS), "repetitions": REPETITIONS,
        "schedule": [f"{t.id}@{e}" for t, e in schedule], "budget_usd": BUDGET_USD,
        "worst_case_run_usd": WORST_CASE_RUN_USD, **checks, "runs": [],
    }
    spent = 0.0
    for index, (task, effort) in enumerate(schedule, 1):
        if spent + WORST_CASE_RUN_USD > BUDGET_USD:
            print(f"STOP: estimated spend ${spent:.4f} + worst case ${WORST_CASE_RUN_USD} would exceed ${BUDGET_USD}")
            manifest["stopped_by_budget"] = True
            break
        summary = run_coding_task(task, make_model(effort), HARNESS, model_settings=settings_for(effort),
                                  runs_dir=out)
        cost = summary.get("estimated_cost_usd")
        if cost is None:  # a dry run makes no API call; an unknown paid cost is assumed to be the worst
            cost_counted = 0.0 if args.dry_run else WORST_CASE_RUN_USD
        else:
            cost_counted = cost
        spent += cost_counted
        manifest["runs"].append(summary["run_id"])
        manifest["estimated_spend_usd"] = round(spent, 6)
        write_run_manifest(out, manifest)  # after every run: a crash still leaves an accurate record
        print(f"[{index:2}/{len(schedule)}] {task.id:20} effort={effort:6} valid={summary['valid']!s:5} "
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
