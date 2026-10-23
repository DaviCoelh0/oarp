"""SciFact loader, download check and safe extraction, on a small dataset in
the same BEIR format (no optional dependency, no network)."""

import hashlib
import io
import shutil
import zipfile
from pathlib import Path

import pytest

from oarp.envs.retrieval import data
from oarp.envs.retrieval.data import DataError, extract_safely, fetch_scifact, load_scifact

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "retrieval"


def copy_fixture(tmp_path: Path) -> Path:
    shutil.copytree(FIXTURE, tmp_path / "data")
    return tmp_path / "data"


def test_train_split_counts_and_skips() -> None:
    split = load_scifact(FIXTURE, "train")
    assert len(split.corpus) == 5 and split.total_queries_in_file == 5
    assert sorted(split.queries) == ["q1", "q2", "q3"]
    assert split.skipped == {"not in queries file": ["q9"], "no relevant document": ["q5"]}
    assert split.qrels["q1"] == {"d1": 1, "d4": 0}  # the explicit 0 is kept, and is not relevant
    assert split.corpus["d1"] == "Aspirin aspirin reduces heart attack risk in adults"


def test_test_split_is_held_out() -> None:
    with pytest.raises(DataError, match="held out"):
        load_scifact(FIXTURE, "test")
    assert sorted(load_scifact(FIXTURE, "test", include_heldout=True).queries) == ["q4"]


def test_unknown_split() -> None:
    with pytest.raises(DataError, match="split must be one of"):
        load_scifact(FIXTURE, "dev")


@pytest.mark.parametrize(("file", "content", "message"), [
    ("corpus.jsonl", '{"_id": "d1", "text": "x"}\n{"_id": "d1", "text": "y"}\n', "duplicate id d1"),
    ("corpus.jsonl", '{"text": "no id"}\n', "malformed record"),
    ("queries.jsonl", "not json\n", "malformed record"),
    ("qrels/train.tsv", "query-id\tcorpus-id\tscore\nq1\td1\n", "expected query-id"),
    ("qrels/train.tsv", "query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td1\t1\n", "duplicate judgment"),
    ("qrels/train.tsv", "query-id\tcorpus-id\tscore\nq1\tmissing_doc\t1\n", "missing from the corpus"),
])
def test_bad_files_are_rejected(tmp_path: Path, file: str, content: str, message: str) -> None:
    root = copy_fixture(tmp_path)
    (root / "scifact" / file).write_text(content)
    with pytest.raises(DataError, match=message):
        load_scifact(root, "train")


def test_missing_data_says_how_to_download(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="python -m oarp.envs.retrieval.data"):
        load_scifact(tmp_path, "train")


def zip_bytes(entries: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, text in entries.items():
            zf.writestr(name, text)
    return buffer.getvalue()


@pytest.mark.parametrize("name", ["../escape.txt", "scifact/../../escape.txt", "/abs/escape.txt"])
def test_unsafe_zip_entries_are_refused(tmp_path: Path, name: str) -> None:
    with pytest.raises(DataError, match="unsafe path"):
        extract_safely(zip_bytes({"scifact/ok.txt": "x", name: "evil"}), tmp_path / "data")
    assert not (tmp_path / "escape.txt").exists() and not (tmp_path / "data" / "scifact").exists()


def archive_url(tmp_path: Path) -> tuple[str, bytes]:
    archive = zip_bytes({f"scifact/{p.relative_to(FIXTURE / 'scifact').as_posix()}": p.read_text()
                         for p in (FIXTURE / "scifact").rglob("*") if p.is_file()})
    path = tmp_path / "scifact.zip"
    path.write_bytes(archive)
    return path.as_uri(), archive


def test_fetch_checks_published_md5_then_pinned_sha256(tmp_path: Path, monkeypatch) -> None:
    url, archive = archive_url(tmp_path)
    monkeypatch.setattr(data, "SCIFACT_MD5", hashlib.md5(archive).hexdigest())
    monkeypatch.setattr(data, "SCIFACT_SHA256", hashlib.sha256(archive).hexdigest())
    identity = fetch_scifact(tmp_path / "data", url=url)
    assert identity["bytes"] == len(archive) and identity["sha256"] == hashlib.sha256(archive).hexdigest()
    assert len(load_scifact(tmp_path / "data", "train").queries) == 3


def test_fetch_refuses_a_wrong_md5(tmp_path: Path, monkeypatch) -> None:
    url, archive = archive_url(tmp_path)
    monkeypatch.setattr(data, "SCIFACT_MD5", "0" * 32)
    with pytest.raises(DataError, match="does not match the published"):
        fetch_scifact(tmp_path / "data", url=url)
    assert not (tmp_path / "data").exists()  # nothing extracted


def test_fetch_refuses_a_wrong_sha256(tmp_path: Path, monkeypatch) -> None:
    url, archive = archive_url(tmp_path)
    monkeypatch.setattr(data, "SCIFACT_MD5", hashlib.md5(archive).hexdigest())
    monkeypatch.setattr(data, "SCIFACT_SHA256", "0" * 64)
    with pytest.raises(DataError, match="does not match the pinned"):
        fetch_scifact(tmp_path / "data", url=url)


def test_real_archive_identity_is_pinned() -> None:
    assert data.SCIFACT_MD5 == "5f7d1de60b170fc8027bb7898e2efca1"
    assert data.SCIFACT_SHA256 == "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"
