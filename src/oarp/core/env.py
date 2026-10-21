"""Read a single value from a dotenv-style file (e.g. the gitignored `.env`).

Deliberately minimal: `NAME=value` or `export NAME=value`, optionally quoted.
The value is returned to the caller only; it is never logged or printed here.
"""

import re
from pathlib import Path


def read_env_value(path: Path, name: str) -> str | None:
    """The value of `name` in `path`, or None if the file or the name is missing."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    pattern = rf"""^\s*(?:export\s+)?{re.escape(name)}\s*=\s*["']?([^"'\s#]+)["']?\s*(?:#.*)?$"""
    match = re.search(pattern, text, re.M)
    return match.group(1) if match else None
