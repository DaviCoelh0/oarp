"""Command-line entry point.

    python -m oarp run --task tasks/coding/dev/off_by_one_001 [--dry-run]
    python -m oarp rescore runs/<run_id>

`run` executes one coding task with Claude Haiku 5.5 (PAID; the same model
settings as the P4 study) or, with --dry-run, with a free scripted model.
`rescore` re-scores a saved run from its final/ files with the independent
verifier and checks that the result matches what was recorded.

Exit codes: 0 = verified success / rescore reproduced, 1 = verified failure /
rescore mismatch, 3 = invalid run (infrastructure or harness error).
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from oarp.core.env import read_env_value
from oarp.core.model import ScriptedModel
from oarp.core.types import ModelResponse, ToolCall
from oarp.envs.coding.runner import HarnessConfig, run_coding_task
from oarp.envs.coding.task import discover_tasks, load_task
from oarp.envs.coding.verify import verify

EXIT_SUCCESS, EXIT_FAILURE, EXIT_INVALID = 0, 1, 3
# Defaults are the P4 study's harness, so a CLI run is comparable to P4 runs.
DEFAULT_HARNESS = HarnessConfig(max_steps=12, max_tool_calls_per_step=1, run_tests_enabled=True)
WORST_CASE_USD_PER_CALL = 10_000 * 0.10e-6 + 4096 * 0.50e-6  # Haiku 5.5, prompt <= 100K


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.handler(args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oarp", description="Observable Agentic RAG Platform")
    commands = parser.add_subparsers(required=True, metavar="command")

    run = commands.add_parser("run", help="run one coding task and record it")
    run.add_argument("--task", type=Path, required=True, help="task folder containing task.toml")
    run.add_argument("--max-steps", type=int, default=DEFAULT_HARNESS.max_steps)
    run.add_argument("--max-tool-calls-per-step", type=int, default=DEFAULT_HARNESS.max_tool_calls_per_step)
    run.add_argument("--no-run-tests", action="store_true", help="do not offer the run_tests tool")
    run.add_argument("--runs-dir", type=Path, default=Path("runs"), help="where run folders are written")
    run.add_argument("--env-file", type=Path, default=Path(".env"), help="file with ANTHROPIC_API_KEY")
    run.add_argument("--dry-run", action="store_true", help="scripted model: no API calls, no cost")
    run.set_defaults(handler=_run)

    rescore = commands.add_parser("rescore", help="re-verify a saved run from its final files")
    rescore.add_argument("run_dir", type=Path)
    rescore.add_argument("--tasks-dir", type=Path, default=Path("tasks"), help="where to find the run's task")
    rescore.set_defaults(handler=_rescore)
    return parser


def _run(args: argparse.Namespace) -> int:
    task = load_task(args.task)
    harness = HarnessConfig(args.max_steps, args.max_tool_calls_per_step, not args.no_run_tests)

    if args.dry_run:
        model = ScriptedModel([
            ModelResponse(tool_calls=(ToolCall("c1", "run_tests", {}),)),
            ModelResponse(tool_calls=(ToolCall("c2", "finish", {"answer": "dry run: no changes made"}),)),
        ])
        settings = {"model": "scripted-dry-run"}
    else:
        from oarp.providers.anthropic_model import AnthropicModel, AnthropicSettings

        key = read_env_value(args.env_file, "ANTHROPIC_API_KEY")
        if not key:
            print(f"error: ANTHROPIC_API_KEY not found in {args.env_file}", file=sys.stderr)
            return EXIT_INVALID
        model_settings = AnthropicSettings()
        settings = asdict(model_settings)
        model = AnthropicModel(model_settings, api_key=key)
        print(f"PAID run with {model_settings.model}: at most {harness.max_steps} model calls, "
              f"worst case about ${harness.max_steps * WORST_CASE_USD_PER_CALL:.3f} (estimate).")

    summary = run_coding_task(task, model, harness, model_settings=settings, runs_dir=args.runs_dir)
    run_dir = args.runs_dir / summary["run_id"]
    if not summary["valid"]:
        print(f"INVALID run {summary['run_id']}: {summary['invalid_reason']}\nfiles: {run_dir}")
        return EXIT_INVALID
    print(
        f"run {summary['run_id']} on {task.id}\n"
        f"  verified success: {summary['verified_success']} ({summary['verification_status']}, "
        f"tests {summary['test_counts']})\n"
        f"  stop reason:      {summary['stop_reason']} after {summary['model_calls']} model calls, "
        f"{summary['tool_calls_attempted']} tool calls\n"
        f"  tokens in/out:    {summary['input_tokens']}/{summary['output_tokens']}, "
        f"estimated cost ${summary['estimated_cost_usd']}\n"
        f"  files:            {run_dir}/ (trace.jsonl, summary.json, final.diff, final/)"
    )
    return EXIT_SUCCESS if summary["verified_success"] else EXIT_FAILURE


def _rescore(args: argparse.Namespace) -> int:
    summary = json.loads((args.run_dir / "summary.json").read_text())
    if not summary.get("valid"):
        print(f"run {summary['run_id']} was invalid and was never scored", file=sys.stderr)
        return EXIT_INVALID
    recorded = next(
        json.loads(line) for line in (args.run_dir / "trace.jsonl").read_text().splitlines()
        if json.loads(line)["event"] == "verification"
    )
    tasks = {t.id: t for t in discover_tasks(args.tasks_dir)}
    task = tasks.get(summary["task_id"])
    if task is None:
        print(f"task {summary['task_id']} not found under {args.tasks_dir}", file=sys.stderr)
        return EXIT_INVALID

    files = {rel: (args.run_dir / "final" / rel).read_text() for rel in task.editable}
    result = verify(task, files)
    same_files = result.file_hashes == recorded["file_hashes"]
    same_status = result.status == recorded["status"]
    print(f"rescore {summary['run_id']} ({task.id}): now {result.status}, recorded {recorded['status']}; "
          f"file hashes {'match' if same_files else 'DIFFER'}")
    return EXIT_SUCCESS if same_files and same_status else EXIT_FAILURE
