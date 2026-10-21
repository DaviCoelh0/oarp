from pathlib import Path

import pytest

from oarp.core.env import read_env_value


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("ANTHROPIC_API_KEY=sk-test-123\n", "sk-test-123"),
        ("export ANTHROPIC_API_KEY=sk-test-123\n", "sk-test-123"),
        ('ANTHROPIC_API_KEY="sk-test-123"\n', "sk-test-123"),
        ("ANTHROPIC_API_KEY='sk-test-123'  # comment\n", "sk-test-123"),
        ("OTHER=1\nANTHROPIC_API_KEY = sk-test-123\n", "sk-test-123"),
        ("# ANTHROPIC_API_KEY=commented-out\n", None),
        ("NOT_ANTHROPIC_API_KEY=x\n", None),
        ("", None),
    ],
)
def test_read_env_value(tmp_path: Path, content: str, expected: str | None) -> None:
    env = tmp_path / ".env"
    env.write_text(content)
    assert read_env_value(env, "ANTHROPIC_API_KEY") == expected


def test_missing_file_is_none(tmp_path: Path) -> None:
    assert read_env_value(tmp_path / "absent.env", "ANTHROPIC_API_KEY") is None
