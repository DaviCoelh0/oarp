"""Shared machinery for paid retrieval studies: preflight, budget guard, seeded
ordering and crash-safe records. No model or dataset logic.

- preflight: in paid mode, refuse unless tracked files are committed, nothing
  is untracked under the guarded folders, the results folder is empty and the
  dataset matches its pinned SHA-256.
- BudgetGuard: before each call, refuse if spent + the worst-case call would
  exceed the cap; a call without a cost counts at the worst case; a call that
  costs MORE than the worst case marks the bound as unsafe, and the run stops.
- hash_order / first_of_pair: label-free, deterministic orderings from
  SHA-256(seed + id), independent of input order and of Python's random module.
- write_manifest / append_record: the manifest is rewritten and each record
  appended after every call, so a crash still leaves an accurate record.

The client-side guard uses ESTIMATED costs; the Console spend limit is the
hard ceiling.
"""

import hashlib
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from oarp.core.trace import git_info

DEFAULT_GUARDED = ("src/", "tests/")


def preflight(repo: Path, out: Path, data_dir: Path, pinned_dataset_sha256: str | None, *,
              paid: bool, guarded: Sequence[str] = DEFAULT_GUARDED) -> dict:
    """The git state (always) and, in paid mode, every refusal check."""
    git = git_info(repo)
    if paid:
        if git is None or git["dirty"]:
            raise SystemExit(f"refusing a paid run: tracked files have uncommitted changes ({git})")
        untracked = subprocess.run(["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard"],
                                   capture_output=True, text=True, timeout=10).stdout.splitlines()
        stray = [p for p in untracked if p.startswith(tuple(guarded))]
        if stray:
            raise SystemExit(f"refusing a paid run: untracked files under guarded folders: {stray}")
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{out} is not empty; refusing to mix runs")
        archive = data_dir / "scifact" / "ARCHIVE.json"
        sha = json.loads(archive.read_text()).get("sha256") if archive.is_file() else None
        if sha != pinned_dataset_sha256:
            raise SystemExit(f"refusing a paid run: dataset SHA-256 {sha} is not the pinned {pinned_dataset_sha256}")
    return {"git": git}


class BudgetGuard:
    def __init__(self, cap_usd: float, worst_case_call_usd: float) -> None:
        self.cap, self.worst = cap_usd, worst_case_call_usd
        self.spent = 0.0
        self.calls_above_worst_case = 0

    def can_call(self) -> bool:
        return self.calls_above_worst_case == 0 and self.spent + self.worst <= self.cap

    def add(self, cost: float | None) -> None:
        self.spent += self.worst if cost is None else cost
        if cost is not None and cost > self.worst:
            self.calls_above_worst_case += 1

    def stop_reason(self) -> str | None:
        if self.calls_above_worst_case:
            return "a call cost more than the assumed worst case: the bound is unsafe"
        if self.spent + self.worst > self.cap:
            return f"estimated spend ${self.spent:.4f} + worst case ${self.worst} would exceed ${self.cap}"
        return None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_order(seed: str, ids: Sequence[str]) -> list[str]:
    return sorted(set(ids), key=lambda i: sha256_text(seed + i))


def first_of_pair(seed: str, item_id: str, pair: tuple[str, str]) -> tuple[str, str]:
    """The pair in the order given by the first bit of SHA-256(seed + id)."""
    return pair if hashlib.sha256((seed + item_id).encode()).digest()[0] < 128 else (pair[1], pair[0])


def write_manifest(out: Path, manifest: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
