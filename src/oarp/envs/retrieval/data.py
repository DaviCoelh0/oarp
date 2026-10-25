"""SciFact in BEIR format: download (pinned, checksummed) and loading.

    <data_dir>/scifact/
      corpus.jsonl      {"_id", "title", "text"} per document
      queries.jsonl     {"_id", "text"} per query (all splits in one file)
      qrels/<split>.tsv query-id, corpus-id, score (header line first)

The split a query belongs to comes only from the qrels file. "train" is used
for development; "test" is held out and is loaded only with an explicit
`include_heldout=True`. Retrieved documents that have no judgment are treated
as non-relevant when scoring (the standard BEIR convention, not proof that
they are irrelevant).

The data is licensed separately (SciFact claims and annotations: CC BY 4.0;
abstracts: ODC-By 1.0) and is never committed: it lives in a gitignored
data/ folder.

    venv/bin/python -m oarp.envs.retrieval.data [data_dir]   # download + verify
"""

import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
SCIFACT_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"  # published in the BEIR repository's dataset table
# Recorded from the first download, which matched the published MD5 (2026-10-22).
SCIFACT_SHA256: str | None = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[4] / "data"
SPLITS = ("train", "test")
HELDOUT_SPLIT = "test"


class DataError(ValueError):
    """The dataset files are missing, malformed or inconsistent."""


@dataclass(frozen=True)
class RetrievalSplit:
    split: str
    corpus: dict[str, str]  # doc id -> "title text"
    queries: dict[str, str]  # query id -> text, only queries with at least one relevant document
    qrels: dict[str, dict[str, int]]  # query id -> {doc id: grade}, for those queries
    skipped: dict[str, list[str]] = field(default_factory=dict)  # reason -> query ids left out
    total_queries_in_file: int = 0


def fetch_scifact(data_dir: Path = DEFAULT_DATA_DIR, url: str = SCIFACT_URL) -> dict:
    """Download the BEIR archive, check it against the published MD5 (and the
    pinned SHA-256 once set), and extract it safely. Returns the archive identity."""
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 (pinned URL)
        archive = response.read()
    md5 = hashlib.md5(archive).hexdigest()  # noqa: S324 (a published fingerprint, not a security boundary)
    sha256 = hashlib.sha256(archive).hexdigest()
    if md5 != SCIFACT_MD5:
        raise DataError(f"archive MD5 {md5} does not match the published {SCIFACT_MD5}")
    if SCIFACT_SHA256 is not None and sha256 != SCIFACT_SHA256:
        raise DataError(f"archive SHA-256 {sha256} does not match the pinned {SCIFACT_SHA256}")
    extract_safely(archive, data_dir)
    identity = {"url": url, "bytes": len(archive), "md5": md5, "sha256": sha256}
    (data_dir / "scifact" / "ARCHIVE.json").write_text(json.dumps(identity, indent=2) + "\n")
    return identity


def extract_safely(archive: bytes, data_dir: Path) -> None:
    """Extract a zip, refusing any entry that would land outside data_dir."""
    data_dir.mkdir(parents=True, exist_ok=True)
    root = data_dir.resolve()
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        for info in zf.infolist():
            name = PurePosixPath(info.filename)
            target = (root / name).resolve()
            if name.is_absolute() or ".." in name.parts or not target.is_relative_to(root):
                raise DataError(f"unsafe path in archive: {info.filename}")
        zf.extractall(root)


def load_scifact(data_dir: Path = DEFAULT_DATA_DIR, split: str = "train", *,
                 include_heldout: bool = False) -> RetrievalSplit:
    if split not in SPLITS:
        raise DataError(f"split must be one of {SPLITS}")
    if split == HELDOUT_SPLIT and not include_heldout:
        raise DataError("the test split is held out; loading it needs include_heldout=True")
    root = data_dir / "scifact"
    corpus = {doc_id: f"{doc.get('title', '')} {doc.get('text', '')}".strip()
              for doc_id, doc in _read_jsonl(root / "corpus.jsonl", "corpus").items()}
    all_queries = {qid: q["text"] for qid, q in _read_jsonl(root / "queries.jsonl", "queries").items()}
    qrels = _read_qrels(root / "qrels" / f"{split}.tsv")

    unknown_docs = sorted({d for judged in qrels.values() for d in judged if d not in corpus})
    if unknown_docs:
        raise DataError(f"qrels point to documents missing from the corpus: {unknown_docs[:5]}")

    skipped: dict[str, list[str]] = {"not in queries file": [], "no relevant document": []}
    kept_qrels = {}
    for qid, judged in sorted(qrels.items()):
        if qid not in all_queries:
            skipped["not in queries file"].append(qid)
        elif not any(grade > 0 for grade in judged.values()):
            skipped["no relevant document"].append(qid)
        else:
            kept_qrels[qid] = judged
    return RetrievalSplit(
        split=split,
        corpus=corpus,
        queries={qid: all_queries[qid] for qid in kept_qrels},
        qrels=kept_qrels,
        skipped={reason: ids for reason, ids in skipped.items() if ids},
        total_queries_in_file=len(all_queries),
    )


def load_titles(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, str]:
    """Doc id -> title alone (the corpus in `load_scifact` joins title and text)."""
    return {doc_id: doc.get("title", "") for doc_id, doc in
            _read_jsonl(data_dir / "scifact" / "corpus.jsonl", "corpus").items()}


def _read_jsonl(path: Path, label: str) -> dict[str, dict]:
    if not path.is_file():
        raise DataError(f"missing {path}; download with: python -m oarp.envs.retrieval.data")
    records: dict[str, dict] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            key = str(record["_id"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise DataError(f"{label} line {number}: malformed record") from exc
        if key in records:
            raise DataError(f"{label} line {number}: duplicate id {key}")
        records[key] = record
    return records


def _read_qrels(path: Path) -> dict[str, dict[str, int]]:
    if not path.is_file():
        raise DataError(f"missing {path}")
    qrels: dict[str, dict[str, int]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines[1:], 2):  # line 1 is the header
        if not line.strip():
            continue
        parts = line.split("\t")
        try:
            qid, doc_id, grade = parts[0], parts[1], int(parts[2])
        except (IndexError, ValueError) as exc:
            raise DataError(f"{path.name} line {number}: expected query-id, corpus-id, score") from exc
        if doc_id in qrels.setdefault(qid, {}):
            raise DataError(f"{path.name} line {number}: duplicate judgment {qid}/{doc_id}")
        qrels[qid][doc_id] = grade
    return qrels


DEV_PARTITION_SEED = "oarp-r2-dev-partition-v1"


def dev_partition(query_ids: list[str], seed: str = DEV_PARTITION_SEED) -> dict[str, list[str]]:
    """Split development query ids into "tune" (choose settings) and "check"
    (confirm a chosen setting once), half each (tune gets the extra one).

    Order = SHA-256 of "<seed>:<query id>", so the split depends only on the
    ids and the seed (not on Python's random module or the input order). Uses
    no labels.
    """
    ordered = sorted(set(query_ids), key=lambda q: hashlib.sha256(f"{seed}:{q}".encode()).hexdigest())
    cut = (len(ordered) + 1) // 2
    return {"tune": sorted(ordered[:cut]), "check": sorted(ordered[cut:])}


if __name__ == "__main__":
    print(json.dumps(fetch_scifact(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_DIR), indent=2))
