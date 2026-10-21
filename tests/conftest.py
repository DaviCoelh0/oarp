from pathlib import Path

import pytest

from oarp.envs.coding import sandbox

TASKS_DIR = Path(__file__).resolve().parents[1] / "tasks"


def _docker_ready() -> bool:
    try:
        sandbox.image_id()
        return True
    except sandbox.SandboxUnavailable:
        return False


requires_docker = pytest.mark.skipif(
    not _docker_ready(),
    reason="Docker daemon or oarp-sandbox image unavailable "
    "(docker build -t oarp-sandbox:0.1 -f docker/sandbox.Dockerfile docker)",
)
