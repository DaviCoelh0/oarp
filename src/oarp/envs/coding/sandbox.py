"""Run untrusted code (task tests over model-written source) in a disposable,
restricted Docker container.

Trust boundary: the host process (agent loop, model calls, API key) is trusted;
everything inside the container is not. Only explicit read-only mounts and a
fixed command go in; only the exit code, capped output and timing come out.

Docker reduces risk but is not a perfect security boundary: containers share
the Docker Desktop VM's Linux kernel, and on macOS that VM can see the folders
Docker Desktop shares (by default /Users). Treat this as protection against
accidental damage and score-gaming by buggy code, not against a skilled attacker.
"""

import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from oarp.core.tools import InfrastructureError

IMAGE = "oarp-sandbox:0.1"  # built from docker/sandbox.Dockerfile
UNPRIVILEGED_USER = "65534:65534"  # "nobody"
DOCKER_RUN_ERROR = 125  # docker itself failed (bad flag, missing image, ...)


class SandboxUnavailable(InfrastructureError):
    """Infrastructure failure (Docker not installed/running, image missing).
    Not the model's fault: a run that hits this is invalid, not failed."""


@dataclass(frozen=True)
class Mount:
    """A host path visible inside the container. Always read-only."""

    source: Path
    target: str


@dataclass(frozen=True)
class SandboxLimits:
    cpus: float = 1.0
    memory_mb: int = 512
    pids: int = 128
    tmpfs_mb: int = 64
    output_bytes: int = 8192  # per stream


@dataclass(frozen=True)
class SandboxResult:
    exit_code: int | None  # None if the run timed out; 137 usually means OOM-killed
    stdout: str
    stderr: str
    timed_out: bool
    truncated: bool
    duration_s: float
    image_id: str = field(default="")  # exact image the code ran in


def docker_command(
    name: str,
    command: list[str],
    mounts: list[Mount],
    limits: SandboxLimits,
    *,
    image: str = IMAGE,
    workdir: str = "/work",
) -> list[str]:
    """The full `docker run` invocation. Kept separate so tests can check every flag."""
    cmd = [
        "docker", "run", "--rm", "--name", name, "--label", "oarp.sandbox=1",
        "--pull", "never",  # never download at run time
        "--init",  # tiny init process: forwards signals, reaps zombies
        "--network", "none",
        "--user", UNPRIVILEGED_USER,
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--read-only",
        "--tmpfs", f"/tmp:rw,size={limits.tmpfs_mb}m,noexec,nosuid,nodev",
        "--memory", f"{limits.memory_mb}m",
        "--memory-swap", f"{limits.memory_mb}m",  # equal to memory: no swap
        "--cpus", str(limits.cpus),
        "--pids-limit", str(limits.pids),
        # The container gets only these variables; the host environment
        # (including any API key) is never passed through.
        "--env", "HOME=/tmp",
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--env", "PYTHONUNBUFFERED=1",
        "--workdir", workdir,
    ]
    for m in mounts:
        cmd += ["--mount", f"type=bind,source={m.source},target={m.target},readonly"]
    return [*cmd, image, *command]


def image_id(image: str = IMAGE) -> str:
    """Return the image ID, or raise SandboxUnavailable if Docker or the image is missing."""
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True, text=True, timeout=30,
        )
    except FileNotFoundError as exc:
        raise SandboxUnavailable("docker CLI not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise SandboxUnavailable("docker did not respond") from exc
    if proc.returncode != 0:
        raise SandboxUnavailable(
            f"docker daemon or image '{image}' unavailable: {proc.stderr.strip()[:300]}"
        )
    return proc.stdout.strip()


def run_in_sandbox(
    command: list[str],
    mounts: list[Mount],
    *,
    timeout_s: float = 30.0,
    limits: SandboxLimits = SandboxLimits(),
    image: str = IMAGE,
    workdir: str = "/work",
) -> SandboxResult:
    for m in mounts:
        if not m.source.is_absolute() or not m.source.exists():
            raise ValueError(f"mount source must be an existing absolute path: {m.source}")
        if "," in str(m.source) or "," in m.target:
            raise ValueError("mount paths must not contain ',' (it separates --mount fields)")

    resolved_image = image_id(image)
    name = f"oarp-{uuid.uuid4().hex[:12]}"
    cmd = docker_command(name, command, mounts, limits, image=image, workdir=workdir)

    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        # Killing the `docker run` client does not stop the container: the
        # daemon owns it. Remove it explicitly by name.
        _force_remove(name)
        stdout, t1 = _cap(exc.stdout, limits.output_bytes)
        stderr, t2 = _cap(exc.stderr, limits.output_bytes)
        return SandboxResult(None, stdout, stderr, True, t1 or t2,
                             time.perf_counter() - started, resolved_image)
    duration = time.perf_counter() - started

    if proc.returncode == DOCKER_RUN_ERROR:
        raise SandboxUnavailable(f"docker run failed: {proc.stderr.decode(errors='replace')[:300]}")
    stdout, t1 = _cap(proc.stdout, limits.output_bytes)
    stderr, t2 = _cap(proc.stderr, limits.output_bytes)
    return SandboxResult(proc.returncode, stdout, stderr, False, t1 or t2, duration, resolved_image)


def _force_remove(name: str) -> None:
    subprocess.run(["docker", "rm", "--force", name], capture_output=True, timeout=30)


def _cap(data: bytes | None, limit: int) -> tuple[str, bool]:
    """Decode and keep the head and tail of oversized output (test summaries are
    at the end, the first failure near the start)."""
    data = data or b""
    if len(data) <= limit:
        return data.decode(errors="replace"), False
    half = limit // 2
    skipped = len(data) - 2 * half
    return (
        data[:half].decode(errors="replace")
        + f"\n... [{skipped} bytes truncated] ...\n"
        + data[-half:].decode(errors="replace"),
        True,
    )
