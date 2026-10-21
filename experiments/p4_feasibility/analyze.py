"""Summarise a P4 results folder into a markdown report (results/analysis.md).

    venv/bin/python experiments/p4_feasibility/analyze.py [results_dir]

Everything is computed from the saved summary.json and trace.jsonl files, so
the report can be regenerated at any time without re-running the agent.
Success rates come with Wilson 95% intervals: with 3 runs per task they are
wide, which is the point: this is a feasibility study, not a benchmark.
"""

import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_runs(results: Path) -> list[dict]:
    runs = []
    for summary_path in sorted(results.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        trace = [json.loads(line) for line in (summary_path.parent / "trace.jsonl").read_text().splitlines()]
        runs.append({"summary": summary, "trace": trace, "dir": summary_path.parent})
    return runs


def recoveries(trace: list[dict]) -> int:
    """Error results followed later by a successful call of the same tool."""
    results = [e for e in trace if e["event"] == "tool_result"]
    count = 0
    for i, event in enumerate(results):
        if event["is_error"] and any(r["name"] == event["name"] and not r["is_error"] for r in results[i + 1:]):
            count += 1
    return count


def median(values: list[float]) -> str:
    return f"{statistics.median(values):g}" if values else "-"


def report(runs: list[dict]) -> str:
    valid = [r for r in runs if r["summary"].get("valid")]
    invalid = [r for r in runs if not r["summary"].get("valid")]
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        by_task[r["summary"]["task_id"]].append(r)

    lines = ["# P4 feasibility results", "",
             f"Runs: {len(runs)} ({len(valid)} valid, {len(invalid)} invalid). "
             "Preliminary feasibility study: 5 dev tasks, small n; not a benchmark.", ""]
    total_cost = sum(r["summary"].get("estimated_cost_usd") or 0 for r in runs)
    successes = sum(r["summary"]["verified_success"] for r in valid)
    low, high = wilson(successes, len(valid))
    lines += [f"**Overall verified success: {successes}/{len(valid)}** "
              f"(Wilson 95% interval {low:.2f}-{high:.2f}). Estimated cost: ${total_cost:.4f}.", ""]

    lines += ["| Task | Success | 95% CI | Stop reasons | Median model calls | Median tool calls | "
              "Used tests | Errors | Recovered | Mean in/out tokens | Cost | Median wall s |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for task_id in sorted(by_task):
        rs = [r["summary"] for r in by_task[task_id]]
        k, n = sum(s["verified_success"] for s in rs), len(rs)
        lo, hi = wilson(k, n)
        stops = Counter(s["stop_reason"] for s in rs)
        errors = Counter()
        for s in rs:
            errors.update(s["errors_by_kind"])
        lines.append(
            f"| {task_id} | {k}/{n} | {lo:.2f}-{hi:.2f} | "
            + ", ".join(f"{s} x{c}" for s, c in stops.items())
            + f" | {median([s['model_calls'] for s in rs])} | {median([s['tool_calls_attempted'] for s in rs])} | "
            f"{sum(s['used_test_feedback'] for s in rs)}/{n} | "
            + (", ".join(f"{e} x{c}" for e, c in errors.items()) or "none")
            + f" | {sum(recoveries(r['trace']) for r in by_task[task_id])} | "
            f"{statistics.mean(s['input_tokens'] for s in rs):.0f}/{statistics.mean(s['output_tokens'] for s in rs):.0f} | "
            f"${sum(s.get('estimated_cost_usd') or 0 for s in rs):.4f} | {median([s['wall_time_s'] for s in rs])} |"
        )

    summaries = sum(1 for r in valid for e in r["trace"] if e["event"] == "model_call" and e.get("reasoning_summary"))
    calls = sum(1 for r in valid for e in r["trace"] if e["event"] == "model_call")
    lines += ["", f"Model calls with a reasoning summary: {summaries}/{calls}.", ""]

    lines += ["## Failures", ""]
    failures = [r for r in valid if not r["summary"]["verified_success"]]
    if not failures:
        lines.append("None.")
    for r in failures:
        s = r["summary"]
        last_text = next((e["text"] for e in reversed(r["trace"]) if e["event"] == "model_call" and e["text"]), "")
        lines += [f"- `{s['run_id']}` {s['task_id']}: stop={s['stop_reason']}, verification={s['verification_status']}, "
                  f"counts={s['test_counts']}, errors={s['errors_by_kind'] or 'none'}",
                  f"  - final answer: {s['final_answer']!r}" + (f"; last text: {last_text[:200]!r}" if last_text else "")]

    lines += ["", "## Passing runs: diffs to review by hand (gaming check)", ""]
    for r in valid:
        if r["summary"]["verified_success"]:
            lines.append(f"- `{r['summary']['run_id']}` {r['summary']['task_id']}: `{r['dir'].name}/final.diff`")

    if invalid:
        lines += ["", "## Invalid runs (not scored)", ""]
        lines += [f"- `{r['summary']['run_id']}` {r['summary']['task_id']}: {r['summary'].get('invalid_reason')}" for r in invalid]
    return "\n".join(lines) + "\n"


def main() -> None:
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS
    text = report(load_runs(results))
    (results / "analysis.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
