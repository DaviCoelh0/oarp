"""Sandbox tests.

Unit tests check the `docker run` invocation and need no Docker. Tests marked
`docker` run real containers and are skipped if Docker or the image is missing
(build it with: docker build -t oarp-sandbox:0.1 -f docker/sandbox.Dockerfile docker).
"""

import subprocess
from pathlib import Path

import pytest
from conftest import requires_docker

from oarp.envs.coding import sandbox
from oarp.envs.coding.sandbox import (
    Mount,
    SandboxLimits,
    SandboxUnavailable,
    docker_command,
    run_in_sandbox,
)

# --- unit tests (no Docker) ---------------------------------------------------------


def test_docker_command_has_every_restriction() -> None:
    cmd = docker_command("oarp-x", ["python", "-V"], [Mount(Path("/a"), "/work")], SandboxLimits())
    joined = " ".join(cmd)
    for flag in [
        "--rm", "--network none", "--user 65534:65534", "--cap-drop ALL",
        "--security-opt no-new-privileges", "--read-only", "--memory 512m",
        "--memory-swap 512m", "--cpus 1.0", "--pids-limit 128", "--pull never",
    ]:
        assert flag in joined, flag
    assert "type=bind,source=/a,target=/work,readonly" in joined
    assert cmd[-3:] == [sandbox.IMAGE, "python", "-V"]


def test_docker_command_never_mounts_socket_or_privileged() -> None:
    cmd = docker_command("n", ["true"], [], SandboxLimits())
    assert "--privileged" not in cmd
    assert not any("docker.sock" in part for part in cmd)
    # Only the three variables we set explicitly; nothing from the host.
    env_values = [cmd[i + 1] for i, part in enumerate(cmd) if part == "--env"]
    assert env_values == ["HOME=/tmp", "PYTHONDONTWRITEBYTECODE=1", "PYTHONUNBUFFERED=1"]


@pytest.mark.parametrize("source", [Path("relative/dir"), Path("/definitely/not/here")])
def test_bad_mount_source_rejected(source: Path) -> None:
    with pytest.raises(ValueError, match="mount source"):
        run_in_sandbox(["true"], [Mount(source, "/work")])


def test_missing_docker_cli_is_infrastructure_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(SandboxUnavailable, match="docker CLI not found"):
        run_in_sandbox(["true"], [])


def test_output_cap_keeps_head_and_tail() -> None:
    text, truncated = sandbox._cap(b"A" * 100 + b"B" * 100, 40)
    assert truncated
    assert text.startswith("A" * 20) and text.endswith("B" * 20)
    assert "[160 bytes truncated]" in text
    assert sandbox._cap(b"short", 40) == ("short", False)


# --- integration tests (real containers) -------------------------------------------


docker = requires_docker


def py(code: str, mounts: list[Mount] | None = None, **kwargs) -> sandbox.SandboxResult:
    return run_in_sandbox(["python", "-c", code], mounts or [], **kwargs)


@pytest.mark.docker
@docker
def test_runs_command_and_reports_exit_code() -> None:
    ok = py("print('hello')")
    assert (ok.exit_code, ok.stdout.strip(), ok.timed_out) == (0, "hello", False)
    assert ok.image_id.startswith("sha256:")
    assert py("raise SystemExit(3)").exit_code == 3


@pytest.mark.docker
@docker
def test_runs_as_nobody() -> None:
    assert py("import os; print(os.getuid(), os.getgid())").stdout.split() == ["65534", "65534"]


@pytest.mark.docker
@docker
def test_no_network() -> None:
    result = py(
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=3); print('CONNECTED')\n"
        "except OSError as e:\n"
        "    print('BLOCKED', type(e).__name__)"
    )
    assert result.stdout.startswith("BLOCKED")


@pytest.mark.docker
@docker
def test_filesystem_is_read_only_except_tmp(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_text("x = 1\n")
    code = (
        "import pathlib\n"
        "for p in ['/work/code.py', '/work/new.py', '/etc/evil', '/tmp/ok']:\n"
        "    try:\n"
        "        pathlib.Path(p).write_text('pwned'); print(p, 'WRITABLE')\n"
        "    except OSError:\n"
        "        print(p, 'DENIED')"
    )
    lines = py(code, [Mount(tmp_path, "/work")]).stdout.splitlines()
    assert lines == ["/work/code.py DENIED", "/work/new.py DENIED", "/etc/evil DENIED", "/tmp/ok WRITABLE"]
    assert (tmp_path / "code.py").read_text() == "x = 1\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["code.py"]


@pytest.mark.docker
@docker
def test_host_secrets_do_not_enter_container(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-FAKE-TEST-VALUE")
    result = py("import os; print(os.environ.get('ANTHROPIC_API_KEY'), sorted(os.environ))")
    assert result.stdout.startswith("None")
    assert "FAKE" not in result.stdout


@pytest.mark.docker
@docker
def test_timeout_kills_and_removes_container() -> None:
    result = py("import time; print('started', flush=True); time.sleep(60)", timeout_s=3)
    assert result.timed_out and result.exit_code is None
    assert result.duration_s < 15
    leftover = subprocess.run(
        ["docker", "ps", "-aq", "--filter", "label=oarp.sandbox=1"], capture_output=True, text=True
    )
    assert leftover.stdout.strip() == ""


@pytest.mark.docker
@docker
def test_process_limit_stops_runaway_spawning() -> None:
    code = (
        "import threading, time\n"
        "n = 0\n"
        "try:\n"
        "    for _ in range(500):\n"
        "        threading.Thread(target=time.sleep, args=(5,), daemon=True).start(); n += 1\n"
        "except RuntimeError:\n"
        "    pass\n"
        "print(n)"
    )
    started = int(py(code, timeout_s=20).stdout.strip())
    assert started < 128


@pytest.mark.docker
@docker
def test_memory_limit_kills_large_allocation() -> None:
    result = py("b = bytearray(1024 * 1024 * 1024); print('ALLOCATED')", timeout_s=20)
    assert result.exit_code != 0
    assert "ALLOCATED" not in result.stdout


@pytest.mark.docker
@docker
def test_large_output_is_truncated() -> None:
    result = py("print('x' * 100_000)", limits=SandboxLimits(output_bytes=1000))
    assert result.truncated
    assert len(result.stdout) < 1100
