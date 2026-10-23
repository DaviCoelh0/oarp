"""S1: listwise reranking of BM25's top k by a model, in one call per query.

The model sees the query and the top-k candidate texts, numbered 1..k (never
document ids or labels), and answers with one line:

    RANKING: 3, 1, 2, ...

Harness settings (recorded with every result): k, the per-document character
cut, the prompt, the repair rule below, and `shuffle`. The ordering itself is
model behaviour.

Shuffle (off by default): the top-k abstracts are shown in a per-query
permutation instead of BM25 order, still numbered 1..k, so an answer that
copies the presentation order no longer reproduces BM25's ranking. The
permutation depends only on the query id and k: position i gets the key
SHA-256(SHA-256(SHUFFLE_SEED + query id) || i), positions sorted by key. The
model's numbers are mapped back through it before becoming document ids.

Repair rule for the RANKING line (the last one in the answer):
- numbers outside 1..k are dropped (counted as `out_of_range`);
- repeated numbers keep their first position (counted as `duplicates`);
- candidates the model left out are appended in BM25 order (`missing`),
  with or without shuffling.
Status: `valid` (a full permutation), `repaired` (at least one valid number,
some repair), `failed` (no RANKING line or no valid number: BM25 order kept),
`model_error` (the client raised: BM25 order kept; the paid runner decides
whether that query is invalid). Documents below rank k keep their BM25 order
after the reranked block, so every result is a full ranking.
"""

import hashlib
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from oarp.core.model import ModelClient
from oarp.core.types import Message, Usage

RERANK_SYSTEM = (
    "You rank scientific abstracts by how relevant they are to a claim. An abstract is relevant if it "
    "contains evidence that supports or refutes the claim."
)
RERANK_INSTRUCTION = (
    "Rank all {k} abstracts from most to least relevant to the claim. Reply with exactly one line: "
    "RANKING: followed by all abstract numbers, most relevant first, separated by commas."
)
SHUFFLE_SEED = "oarp-r2-rerank-shuffle-v1:"
_RANKING_LINE = re.compile(r"^\s*RANKING:\s*(.*)$", re.I | re.M)


@dataclass(frozen=True)
class RerankConfig:
    """Harness settings of S1; recorded with every result."""

    k: int = 20
    max_doc_chars: int = 2000  # SciFact p90 is ~2,100 characters, so most abstracts are whole
    shuffle: bool = False  # present the top k in a per-query seeded permutation


@dataclass(frozen=True)
class ParsedRanking:
    order: list[int]  # 0-based positions into the candidate list, best first, length k
    status: str  # valid | repaired | failed
    duplicates: int = 0
    out_of_range: int = 0
    missing: int = 0


@dataclass(frozen=True)
class RerankRecord:
    ranking: list[str]  # full ranking: reranked top k, then BM25's remainder
    status: str  # valid | repaired | failed | model_error
    duplicates: int
    out_of_range: int
    missing: int
    usage: Usage | None
    model: str | None
    latency_s: float
    copied_presentation_order: bool  # the answer was exactly 1..k in presentation order
    matches_bm25_order: bool  # the final top k equals BM25's top k
    error: str | None = None


def truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + " …"


def numbered(texts: Sequence[str], max_chars: int) -> str:
    return "\n\n".join(f"[{i}] {truncate(t, max_chars)}" for i, t in enumerate(texts, 1))


def rerank_messages(query: str, texts: Sequence[str], config: RerankConfig) -> list[Message]:
    """The whole request: only the query and candidate texts, never ids or labels."""
    user = (f"Claim: {query}\n\nAbstracts:\n\n{numbered(texts, config.max_doc_chars)}\n\n"
            + RERANK_INSTRUCTION.format(k=len(texts)))
    return [Message("system", RERANK_SYSTEM), Message("user", user)]


def parse_ranking(text: str, k: int) -> ParsedRanking:
    lines = _RANKING_LINE.findall(text)
    numbers = [int(n) for n in re.findall(r"\d+", lines[-1])] if lines else []
    order: list[int] = []
    duplicates = out_of_range = 0
    for n in numbers:
        if not 1 <= n <= k:
            out_of_range += 1
        elif n - 1 in order:
            duplicates += 1
        else:
            order.append(n - 1)
    if not order:
        return ParsedRanking(list(range(k)), "failed", duplicates, out_of_range, k)
    missing = [i for i in range(k) if i not in order]
    status = "valid" if not (duplicates or out_of_range or missing) else "repaired"
    return ParsedRanking(order + missing, status, duplicates, out_of_range, len(missing))


def presentation_order(query_id: str, k: int) -> list[int]:
    """BM25 positions in the order they are shown: shown number j + 1 is BM25 position result[j]."""
    seed = hashlib.sha256(f"{SHUFFLE_SEED}{query_id}".encode()).digest()
    return sorted(range(k), key=lambda i: hashlib.sha256(seed + i.to_bytes(4, "big")).digest())


def rerank(model: ModelClient, query_id: str, query: str, ranking: Sequence[str], corpus: dict[str, str],
           config: RerankConfig = RerankConfig()) -> RerankRecord:
    """Rerank the top k of a full BM25 ranking (doc ids, best first)."""
    top = list(ranking[: config.k])
    rest = list(ranking[config.k:])
    shown = presentation_order(query_id, len(top)) if config.shuffle else list(range(len(top)))
    started = time.perf_counter()
    try:
        response = model.complete(rerank_messages(query, [corpus[top[i]] for i in shown], config), [])
    except Exception as exc:  # the paid runner decides whether such a query is invalid
        return RerankRecord(list(ranking), "model_error", 0, 0, 0, None, None, time.perf_counter() - started,
                            False, True, f"{type(exc).__name__}: {exc}")
    latency = time.perf_counter() - started
    parsed = parse_ranking(response.text, len(top))
    named = parsed.order[: len(top) - parsed.missing]  # shown positions the model named, in its order
    bm25_positions = [shown[p] for p in named]
    bm25_positions += sorted(set(range(len(top))) - set(bm25_positions))  # missing: BM25 order
    reranked = [top[i] for i in bm25_positions]
    copied = parsed.status == "valid" and parsed.order == list(range(len(top)))
    return RerankRecord(reranked + rest, parsed.status, parsed.duplicates, parsed.out_of_range, parsed.missing,
                        response.usage, response.model, latency, copied, reranked == top)


def provenance(config: RerankConfig) -> dict:
    return {"rerank_config": asdict(config), "rerank_shuffle_seed": SHUFFLE_SEED if config.shuffle else None,
            "rerank_system": RERANK_SYSTEM, "rerank_instruction": RERANK_INSTRUCTION}
