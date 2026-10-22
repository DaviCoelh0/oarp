"""Summarise an M2 pilot results folder into results/analysis.md.

    venv/bin/python experiments/m2_pilot/analyze.py [results_dir]

Computed from the saved summary.json and trace.jsonl files only (plus the
verifier, in the sandbox, for the replayed candidate), so the report can be
regenerated at any time without calling the model. Definitions, kept apart
on purpose:

- tests run: successful run_tests calls;
- failing results seen: run_tests results whose exit code was not 0;
- edits after a failure: successful write_file calls made after the first
  failing result was seen;
- first tested candidate: the editable files at the moment of the first
  run_tests call after at least one write_file (the state the agent chose to
  test), re-scored by the independent verifier. Runs that never test after
  writing have none.

Hidden-test output is never written here: only statuses and counts.
"""

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from oarp.envs.coding.task import CodingTask, discover_tasks, task_hash  # noqa: E402
from oarp.envs.coding.verify import verify  # noqa: E402

DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"
TASKS_DIR = REPO / "tasks"
EXIT_CODE = re.compile(r"^Test command exited with code (-?\d+)")


def load_runs(results: Path) -> list[dict]:
    runs = []
    for summary_path in sorted(results.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        trace = [json.loads(line) for line in (summary_path.parent / "trace.jsonl").read_text().splitlines()]
        runs.append({"summary": summary, "trace": trace, "dir": summary_path.parent})
    return runs


def tool_results(trace: list[dict]) -> list[dict]:
    return [e for e in trace if e["event"] == "tool_result"]


def test_exit_code(content: str) -> int | None:
    """Exit code of a run_tests result; None for a timeout."""
    match = EXIT_CODE.match(content)
    return int(match.group(1)) if match else None


def feedback_metrics(trace: list[dict]) -> dict:
    tests_run = failing_seen = edits_after_failure = writes = 0
    for event in tool_results(trace):
        if event["is_error"]:
            continue
        if event["name"] == "run_tests":
            tests_run += 1
            failing_seen += test_exit_code(event["content"]) != 0
        elif event["name"] == "write_file":
            writes += 1
            edits_after_failure += failing_seen > 0
    return {"tests_run": tests_run, "failing_seen": failing_seen,
            "edits_after_failure": edits_after_failure, "writes": writes}


def first_tested_candidate(task: CodingTask, trace: list[dict]) -> dict[str, str] | None:
    files = {rel: (task.repo_dir / rel).read_text() for rel in task.editable}
    written = False
    for event in tool_results(trace):
        if event["is_error"]:
            continue
        if event["name"] == "write_file":
            files[event["arguments"]["path"]] = event["arguments"]["content"]
            written = True
        elif event["name"] == "run_tests" and written:
            return dict(files)
    return None


def analyse_run(run: dict, tasks: dict[str, CodingTask]) -> dict:
    s = run["summary"]
    task = tasks[s["task_id"]]
    if task_hash(task) != s["task_hash"]:
        raise SystemExit(f"{s['run_id']}: task {task.id} changed since the run")
    row = {**feedback_metrics(run["trace"]), "summary": s, "candidate": None}
    candidate = first_tested_candidate(task, run["trace"])
    if candidate is not None:
        v = verify(task, candidate)
        row["candidate"] = {"status": v.status, "public": v.public.status,
                            "hidden": v.hidden.status if v.hidden else None}
    return row


def report(rows: list[dict], invalid: list[dict]) -> str:
    lines = ["# M2 pilot results", "",
             f"Runs: {len(rows) + len(invalid)} ({len(rows)} valid, {len(invalid)} invalid). "
             "Feasibility only: 2 dev tasks x 3 repetitions, one configuration. "
             "Repetitions of a task are clustered; no interval is given.", ""]
    total = sum(r["summary"].get("estimated_cost_usd") or 0 for r in rows + invalid)
    lines += [f"Estimated cost: ${total:.4f}.", ""]

    lines += ["## Per run", "",
              "| Run | Task | Success | Public | Hidden | Stop | Model calls | Tool calls | Tests run | "
              "Failing seen | Edits after failure | First tested candidate | Errors | In/out tokens | Cost |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s, c = r["summary"], r["candidate"]
        candidate = "none" if c is None else f"{c['status']} (public {c['public']}, hidden {c['hidden']})"
        lines.append(
            f"| `{s['run_id']}` | {s['task_id']} | {s['verified_success']} | {s['public_status']} | "
            f"{s['hidden_status']} | {s['stop_reason']} | {s['model_calls']} | {s['tool_calls_attempted']} | "
            f"{r['tests_run']} | {r['failing_seen']} | {r['edits_after_failure']} | {candidate} | "
            f"{', '.join(f'{k} x{v}' for k, v in s['errors_by_kind'].items()) or 'none'} | "
            f"{s['input_tokens']}/{s['output_tokens']} | ${s.get('estimated_cost_usd')} |")

    lines += ["", "## Per task", "",
              "| Task | Verified success | Passed public, failed hidden | First tested candidate passed | "
              "Recovered after a failing result | Hit max_steps | Median model calls |",
              "|---|---|---|---|---|---|---|"]
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["summary"]["task_id"]].append(r)
    for task_id, rs in sorted(by_task.items()):
        n = len(rs)
        success = sum(r["summary"]["verified_success"] for r in rs)
        public_only = sum(r["summary"]["public_status"] == "passed" and not r["summary"]["verified_success"] for r in rs)
        first_ok = sum(r["candidate"] is not None and r["candidate"]["status"] == "passed" for r in rs)
        recovered = sum(r["summary"]["verified_success"] and r["failing_seen"] > 0 and r["edits_after_failure"] > 0
                        for r in rs)
        max_steps = sum(r["summary"]["stop_reason"] == "max_steps" for r in rs)
        calls = statistics.median(r["summary"]["model_calls"] for r in rs)
        lines.append(f"| {task_id} | {success}/{n} | {public_only}/{n} | {first_ok}/{n} | {recovered}/{n} | "
                     f"{max_steps}/{n} | {calls:g} |")

    lines += ["", "## Failures", ""]
    failures = [r for r in rows if not r["summary"]["verified_success"]]
    lines += ["None."] if not failures else []
    for r in failures:
        s = r["summary"]
        lines.append(f"- `{s['run_id']}` {s['task_id']}: stop={s['stop_reason']}, public={s['public_status']} "
                     f"{s['test_counts']}, hidden={s['hidden_status']} {s['hidden_counts']}; "
                     f"final answer: {s['final_answer']!r}")

    lines += ["", "## Diffs to review by hand", ""]
    lines += [f"- `{r['summary']['run_id']}` {r['summary']['task_id']} "
              f"({'passed' if r['summary']['verified_success'] else 'failed'}): `{r['summary']['run_id']}/final.diff`"
              for r in rows]
    if invalid:
        lines += ["", "## Invalid runs (not scored)", ""]
        lines += [f"- `{r['summary']['run_id']}` {r['summary']['task_id']}: {r['summary'].get('invalid_reason')}"
                  for r in invalid]
    return "\n".join(lines) + "\n"


def main() -> None:
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS
    tasks = {t.id: t for t in discover_tasks(TASKS_DIR)}
    runs = load_runs(results)
    rows = [analyse_run(r, tasks) for r in runs if r["summary"].get("valid")]
    invalid = [r for r in runs if not r["summary"].get("valid")]
    text = report(rows, invalid)
    (results / "analysis.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
