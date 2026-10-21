"""Append-only JSONL traces and run provenance. Environment-independent.

One event per line, flushed immediately, so a crash still leaves every event
up to that point. The tracer adds timing and an estimated cost to model calls;
it never writes environment variables, request headers or credentials.
"""

import hashlib
import json
import secrets
import subprocess
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

from oarp.core.pricing import PRICES_AS_OF, estimate_cost
from oarp.core.types import Usage


def new_run_id() -> str:
    """Sortable and unique: UTC timestamp + random suffix, e.g. 20261021T153012-a1b2c3."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(3)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash(obj: Any) -> str:
    """Hash of a JSON-serialisable object, independent of dict key order."""
    return sha256_text(json.dumps(obj, sort_keys=True, default=_to_json))


def git_info(repo_dir: Path) -> dict[str, Any] | None:
    """Commit and whether the working tree has uncommitted changes (None outside git)."""
    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(repo_dir), *args], capture_output=True, text=True, timeout=10)

    try:
        commit = git("rev-parse", "HEAD")
        status = git("status", "--porcelain")
    except (OSError, subprocess.TimeoutExpired):
        return None
    if commit.returncode != 0:
        return None
    return {"commit": commit.stdout.strip(), "dirty": bool(status.stdout.strip())}


class JsonlTracer:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file: TextIO = path.open("a", encoding="utf-8")
        self._started = time.perf_counter()
        self.total_cost_usd = 0.0
        self.cost_complete = True  # False once any model call had no cost estimate

    def emit(self, event: dict[str, Any]) -> None:
        event = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                 "t_s": round(time.perf_counter() - self._started, 4), **event}
        if event.get("event") == "model_call":
            cost = estimate_cost(event.get("model"), Usage(event.get("input_tokens"), event.get("output_tokens")))
            event["estimated_cost_usd"] = None if cost is None else round(cost, 8)
            event["prices_as_of"] = PRICES_AS_OF
            if cost is None:
                self.cost_complete = False
            else:
                self.total_cost_usd += cost
        self._file.write(json.dumps(event, default=_to_json) + "\n")
        self._file.flush()

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self._started

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "JsonlTracer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _to_json(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")
