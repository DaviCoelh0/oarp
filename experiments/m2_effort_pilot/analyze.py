"""Summarise an M2 effort-pilot results folder into results/analysis.md.

    venv/bin/python experiments/m2_effort_pilot/analyze.py [results_dir]

Reuses the metric definitions of experiments/m2_pilot/analyze.py (tests run,
failing results seen, edits after a failure, first tested candidate) and
groups them by (task, effort). Computed from saved files only, plus the
verifier for the replayed candidate. Hidden-test output is never written here.
"""

import importlib.util
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from oarp.envs.coding.task import discover_tasks  # noqa: E402

_spec = importlib.util.spec_from_file_location("m2_pilot_analyze", REPO / "experiments" / "m2_pilot" / "analyze.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"
TASKS_DIR = REPO / "tasks"


def effort(row: dict) -> str:
    return row["summary"]["config"]["model"].get("effort") or "default"


def tool_sequence(trace: list[dict]) -> str:
    steps = []
    for event in base.tool_results(trace):
        args = event["arguments"]
        step = event["name"] + (f"({args['path']})" if "path" in args else "")
        if event["is_error"]:
            step += "!refused" if event.get("error_kind") == "tool_refused" else "!error"
        elif event["name"] == "run_tests":
            code = base.test_exit_code(event["content"])
            step += "=pass" if code == 0 else f"=FAIL({code})"
        steps.append(step)
    return " > ".join(steps)


def report(rows: list[dict], invalid: list[dict]) -> str:
    total = sum(r["summary"].get("estimated_cost_usd") or 0 for r in rows + invalid)
    lines = ["# M2 effort pilot results", "",
             f"Runs: {len(rows) + len(invalid)} ({len(rows)} valid, {len(invalid)} invalid). "
             "Model-configuration feasibility: effort low vs medium, 2 dev tasks x 3 repetitions each. "
             "Repetitions are clustered; no interval is given.", "", f"Estimated cost: ${total:.4f}.", ""]

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["summary"]["task_id"], effort(r))].append(r)
    lines += ["## Per task and effort", "",
              "| Task | Effort | Verified success | Passed public, failed hidden | First tested candidate passed | "
              "Failing results seen | Recovered after a failure | Hit max_steps | Median model calls | "
              "Median in/out tokens | Median cost | Median wall s |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (task_id, level), rs in sorted(groups.items()):
        n = len(rs)
        s = [r["summary"] for r in rs]
        lines.append(
            f"| {task_id} | {level} | {sum(x['verified_success'] for x in s)}/{n} | "
            f"{sum(x['public_status'] == 'passed' and not x['verified_success'] for x in s)}/{n} | "
            f"{sum(r['candidate'] is not None and r['candidate']['status'] == 'passed' for r in rs)}/{n} | "
            f"{sum(r['failing_seen'] > 0 for r in rs)}/{n} | "
            f"{sum(r['summary']['verified_success'] and r['failing_seen'] > 0 and r['edits_after_failure'] > 0 for r in rs)}/{n} | "
            f"{sum(x['stop_reason'] == 'max_steps' for x in s)}/{n} | "
            f"{statistics.median(x['model_calls'] for x in s):g} | "
            f"{statistics.median(x['input_tokens'] for x in s):.0f}/{statistics.median(x['output_tokens'] for x in s):.0f} | "
            f"${statistics.median(x.get('estimated_cost_usd') or 0 for x in s):.4f} | "
            f"{statistics.median(x['wall_time_s'] for x in s):.1f} |")

    lines += ["", "## Per run", "",
              "| Run | Task | Effort | Success | Public | Hidden | Stop | Calls | First tested candidate | Tool sequence |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["summary"]["run_id"]):
        s, c = r["summary"], r["candidate"]
        candidate = "none" if c is None else c["status"]
        lines.append(f"| `{s['run_id']}` | {s['task_id']} | {effort(r)} | {s['verified_success']} | "
                     f"{s['public_status']} | {s['hidden_status']} | {s['stop_reason']} | {s['model_calls']} | "
                     f"{candidate} | {r['sequence']} |")

    lines += ["", "## Failures", ""]
    failures = [r for r in rows if not r["summary"]["verified_success"]]
    lines += ["None."] if not failures else []
    for r in failures:
        s = r["summary"]
        lines.append(f"- `{s['run_id']}` {s['task_id']} @ {effort(r)}: stop={s['stop_reason']}, "
                     f"public={s['public_status']} {s['test_counts']}, hidden={s['hidden_status']} "
                     f"{s['hidden_counts']}; final answer: {s['final_answer']!r}")
    if invalid:
        lines += ["", "## Invalid runs (not scored)", ""]
        lines += [f"- `{r['summary']['run_id']}` {r['summary']['task_id']}: {r['summary'].get('invalid_reason')}"
                  for r in invalid]
    return "\n".join(lines) + "\n"


def main() -> None:
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS
    tasks = {t.id: t for t in discover_tasks(TASKS_DIR)}
    runs = base.load_runs(results)
    rows = []
    for run in runs:
        if run["summary"].get("valid"):
            row = base.analyse_run(run, tasks)
            row["sequence"] = tool_sequence(run["trace"])
            rows.append(row)
    invalid = [r for r in runs if not r["summary"].get("valid")]
    text = report(rows, invalid)
    (results / "analysis.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
