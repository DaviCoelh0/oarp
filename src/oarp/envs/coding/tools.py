"""Agent-facing tools for the coding environment: thin wrappers that expose a
Workspace and the sandbox through the shared Tool protocol.

- read_file / write_file run on the host and only touch the workspace
  (Workspace enforces the path rules and the write allowlist).
- run_tests is the only tool that executes code, always inside the sandbox.
  A failing test run is NOT a tool error: the tool worked, the code is wrong.
"""

from oarp.core.tools import Tool
from oarp.core.types import ToolSpec
from oarp.envs.coding.sandbox import SandboxLimits, run_in_sandbox
from oarp.envs.coding.workspace import Workspace

READ_LIMIT_BYTES = 20_000

# Deliberately neutral about running tests: whether the agent chooses to use
# test feedback is something we want to measure, not prescribe.
SYSTEM_PROMPT = (
    "You are a software engineering agent working in a small Python repository. "
    "Use the tools to inspect and change files. You can only modify files marked "
    "editable. When you are done, or if you cannot complete the task, call finish "
    "with a short summary. Always end by calling finish."
)


def task_prompt(ws: Workspace) -> str:
    """The task text plus the file list (saves the agent a discovery step)."""
    editable = set(ws.task.editable)
    files = "\n".join(f"- {p} (editable)" if p in editable else f"- {p}" for p in ws.list_files())
    return f"{ws.task.prompt}\n\nFiles:\n{files}"


def coding_tools(
    ws: Workspace, *, run_tests_enabled: bool = True, limits: SandboxLimits = SandboxLimits()
) -> list[Tool]:
    def read_file(path: str) -> str:
        content = ws.read_text(path)
        data = content.encode("utf-8")
        if len(data) <= READ_LIMIT_BYTES:
            return content
        shown = data[:READ_LIMIT_BYTES].decode("utf-8", errors="ignore")
        return f"{shown}\n... [truncated: showing {READ_LIMIT_BYTES} of {len(data)} bytes]"

    def write_file(path: str, content: str) -> str:
        ws.write_text(path, content)
        return f"Wrote {len(content.encode('utf-8'))} bytes to {path}."

    def run_tests() -> str:
        result = run_in_sandbox(
            list(ws.task.test_command), ws.sandbox_mounts(), timeout_s=ws.task.timeout_s, limits=limits
        )
        if result.timed_out:
            header = f"Tests timed out after {ws.task.timeout_s:g}s; the process was killed."
        else:
            header = f"Test command exited with code {result.exit_code}" + (
                " (all tests passed)." if result.exit_code == 0 else "."
            )
            if result.exit_code == 137:
                header += " The process was killed (likely out of memory)."
        parts = [header]
        if result.stdout.strip():
            parts.append(f"--- stdout ---\n{result.stdout.rstrip()}")
        if result.stderr.strip():
            parts.append(f"--- stderr ---\n{result.stderr.rstrip()}")
        return "\n".join(parts)

    path_param = {"type": "string", "description": "Path relative to the repository root, e.g. src/module.py"}
    tools = [
        Tool(
            ToolSpec(
                "read_file",
                "Read a text file from the repository.",
                {"type": "object", "properties": {"path": path_param}, "required": ["path"]},
            ),
            read_file,
        ),
        Tool(
            ToolSpec(
                "write_file",
                "Replace the entire content of an editable file. Provide the complete new file content.",
                {
                    "type": "object",
                    "properties": {"path": path_param, "content": {"type": "string", "description": "Full new file content."}},
                    "required": ["path", "content"],
                },
            ),
            write_file,
        ),
    ]
    if run_tests_enabled:
        tools.append(
            Tool(
                ToolSpec(
                    "run_tests",
                    "Run the repository's test suite and return the exit code and output.",
                    {"type": "object", "properties": {}},
                ),
                run_tests,
            )
        )
    return tools
