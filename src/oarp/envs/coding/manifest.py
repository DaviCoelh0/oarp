"""Frozen task-set versions: a manifest records exactly which tasks, in which
version, an experiment is allowed to use.

    {"tasks": [{"id": ..., "split": ..., "hash": task_hash}, ...], "set_hash": ...}

An experiment script writes the manifest once, before its first run, and checks
it before every batch; any byte changed in any task folder (definition,
starting repo, public or hidden tests, solution, decoys) makes the check fail.
"""

import json
from pathlib import Path
from typing import Any

from oarp.core.trace import stable_hash
from oarp.envs.coding.task import CodingTask, discover_tasks, task_hash


def manifest_for(tasks: list[CodingTask]) -> dict[str, Any]:
    entries = [{"id": t.id, "split": t.split, "hash": task_hash(t)} for t in sorted(tasks, key=lambda t: t.id)]
    return {"tasks": entries, "set_hash": stable_hash(entries)}


def write_manifest(tasks: list[CodingTask], path: Path) -> dict[str, Any]:
    if not tasks:
        raise ValueError("refusing to freeze an empty task set")
    manifest = manifest_for(tasks)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def check_manifest(path: Path, tasks_dir: Path) -> list[str]:
    """Problems found (empty = the task set is exactly as frozen)."""
    manifest = json.loads(path.read_text())
    available = {t.id: t for t in discover_tasks(tasks_dir, include_heldout=True)}
    problems = []
    for entry in manifest["tasks"]:
        task = available.get(entry["id"])
        if task is None:
            problems.append(f"{entry['id']}: missing under {tasks_dir}")
        elif task.split != entry["split"]:
            problems.append(f"{entry['id']}: split is now '{task.split}', frozen as '{entry['split']}'")
        elif task_hash(task) != entry["hash"]:
            problems.append(f"{entry['id']}: changed since the manifest was written")
    if stable_hash(manifest["tasks"]) != manifest["set_hash"]:
        problems.append("the manifest file itself was edited (set_hash mismatch)")
    return problems
