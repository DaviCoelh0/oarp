"""Run one coding task end to end and record it:

    workspace -> tools -> run_agent (traced) -> independent verifier -> summary

Each run writes runs/<run_id>/trace.jsonl (events as they happen),
summary.json (one record for analysis), final.diff and final/<editable files>
(so the run can be re-scored later). Infrastructure failures mark the run
INVALID instead of scoring it as an agent failure.
"""

import json
import platform
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

from oarp.core.loop import run_agent
from oarp.core.model import ModelClient
from oarp.core.tools import InfrastructureError, ToolRegistry
from oarp.core.trace import JsonlTracer, git_info, new_run_id, sha256_text, stable_hash
from oarp.core.types import ErrorKind, RunResult
from oarp.envs.coding.task import CodingTask
from oarp.envs.coding.tools import SYSTEM_PROMPT, coding_tools, task_prompt
from oarp.envs.coding.verify import VerificationResult, final_files, verify
from oarp.envs.coding.workspace import Workspace

REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class HarnessConfig:
    """The harness variables a comparison may change (one at a time)."""

    max_steps: int = 10
    max_tool_calls_per_step: int = 1
    run_tests_enabled: bool = True
    # Which implementation of the same loop runs the episode: "plain" is
    # oarp.core.loop.run_agent, "graph" is the LangGraph version
    # (oarp.core.graph_loop, optional extra). Recorded in the config hash.
    loop: Literal["plain", "graph"] = "plain"


def loop_function(name: str) -> Callable[..., RunResult]:
    """The loop implementation named by HarnessConfig.loop."""
    if name == "plain":
        return run_agent
    if name == "graph":
        try:
            from oarp.core.graph_loop import run_graph_agent
        except ImportError as exc:
            raise ImportError(
                "the graph loop needs the optional extra: pip install -e '.[langgraph]'"
            ) from exc
        return run_graph_agent
    raise ValueError(f"unknown loop implementation: {name!r}")


def run_coding_task(
    task: CodingTask,
    model: ModelClient,
    harness: HarnessConfig,
    *,
    model_settings: dict[str, Any],
    runs_dir: Path,
) -> dict[str, Any]:
    """`model_settings` describes the model as configured (e.g. asdict(AnthropicSettings())),
    so the configuration hash covers both harness and model."""
    run_loop = loop_function(harness.loop)  # fails before any run files are written
    run_id = new_run_id()
    run_dir = runs_dir / run_id
    config = {"harness": asdict(harness), "model": model_settings}
    summary: dict[str, Any] = {
        "run_id": run_id, "task_id": task.id, "split": task.split,
        "config": config, "config_hash": stable_hash(config),
    }

    with JsonlTracer(run_dir / "trace.jsonl") as tracer, Workspace.create(task) as ws:
        tools = coding_tools(ws, run_tests_enabled=harness.run_tests_enabled)
        prompt = task_prompt(ws)
        tracer.emit({
            "event": "run_start", **summary,
            "prompt_hashes": {
                "system": sha256_text(SYSTEM_PROMPT),
                "task": sha256_text(prompt),
                "tools": stable_hash([t.spec for t in tools]),
            },
            "system_prompt": SYSTEM_PROMPT, "task_prompt": prompt,
            "provenance": _provenance(),
        })
        try:
            result = run_loop(
                model, ToolRegistry(tools), system_prompt=SYSTEM_PROMPT, task_prompt=prompt,
                max_steps=harness.max_steps, max_tool_calls_per_step=harness.max_tool_calls_per_step,
                on_event=tracer.emit,
            )
            files = final_files(ws)
            verification = verify(task, files)
        except InfrastructureError as exc:
            reason = f"{type(exc).__name__}: {exc}"
            tracer.emit({"event": "run_invalid", "reason": reason})
            summary.update(valid=False, invalid_reason=reason, estimated_cost_usd=_cost(tracer))
            _write_json(run_dir / "summary.json", summary)
            return summary

        tracer.emit({"event": "verification", **_verification_fields(verification)})
        summary.update(_summarize(result, verification, tracer))

    _save_final_files(run_dir, files, verification)
    _write_json(run_dir / "summary.json", summary)
    return summary


def _summarize(result: RunResult, v: VerificationResult, tracer: JsonlTracer) -> dict[str, Any]:
    results = [r for step in result.steps for r in step.tool_results]
    rejected = [r for r in results if r.error_kind is ErrorKind.OVER_LIMIT]
    attempted = [r for r in results if r.error_kind not in (ErrorKind.OVER_LIMIT, ErrorKind.INVALID_FINISH)]
    test_runs = [r for r in attempted if r.name == "run_tests" and not r.is_error]
    usages = [s.response.usage for s in result.steps if s.response.usage]
    return {
        "valid": v.valid,
        "invalid_reason": None if v.valid else f"verifier {v.status}",
        # Termination and task success are recorded separately, on purpose.
        "stop_reason": result.stop_reason,
        "verified_success": v.passed,
        "verification_status": v.status,
        "test_counts": v.counts,
        "model_calls": len(result.steps),
        "tool_calls_attempted": len(attempted),
        "tool_calls_rejected": len(rejected),
        "errors_by_kind": dict(Counter(r.error_kind.value for r in results if r.error_kind)),
        "run_tests_calls": len(test_runs),
        "used_test_feedback": bool(test_runs),
        "input_tokens": sum(u.input_tokens or 0 for u in usages),
        "output_tokens": sum(u.output_tokens or 0 for u in usages),
        "estimated_cost_usd": _cost(tracer),
        "models_served": sorted({s.response.model for s in result.steps if s.response.model}),
        "final_answer": result.final_answer,
        "error": result.error,
        "wall_time_s": round(tracer.elapsed_s, 3),
        "sandbox_image": v.image_id,
    }


def _cost(tracer: JsonlTracer) -> float | None:
    return round(tracer.total_cost_usd, 6) if tracer.cost_complete else None


def _verification_fields(v: VerificationResult) -> dict[str, Any]:
    return {
        "status": v.status, "passed": v.passed, "valid": v.valid, "exit_code": v.exit_code,
        "counts": v.counts, "file_hashes": v.file_hashes, "duration_s": round(v.duration_s, 3),
        "image_id": v.image_id, "output": v.output,
    }


def _provenance() -> dict[str, Any]:
    def version(package: str) -> str | None:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return None

    return {"git": git_info(REPO_ROOT), "python": platform.python_version(),
            "anthropic_sdk": version("anthropic"), "langgraph": version("langgraph"),
            "platform": platform.platform()}


def _save_final_files(run_dir: Path, files: dict[str, str], v: VerificationResult) -> None:
    (run_dir / "final.diff").write_text(v.diff)
    for rel, content in files.items():
        target = run_dir / "final" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")
