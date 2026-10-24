"""S2b: a model gate that decides, per query, whether to keep BM25's order.

One call per query. The model sees the query and BM25's top 3 results
(fixed, `GATE_TOP`), numbered 1..3, and answers whether result 1 is relevant:

    ANSWER: yes    -> keep the BM25 order
    ANSWER: no     -> use the S1 (reranked) ranking

Harness settings: the prompt, the number of results shown (fixed at 3), the
per-document character cut, and the fallback: an answer without a parseable
ANSWER line counts as "rerank" (`unparseable`, counted). A client error is
recorded as `model_error`, also treated as "rerank"; the paid runner decides
whether such a query is invalid. The yes/no itself is model behaviour.
"""

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass

from oarp.core.model import ModelClient
from oarp.core.types import Message, Usage
from oarp.envs.retrieval.rerank import numbered

GATE_TOP = 3
GATE_MAX_DOC_CHARS = 2000
GATE_SYSTEM = (
    "You judge whether a scientific abstract is relevant to a claim. An abstract is relevant if it "
    "contains evidence that supports or refutes the claim."
)
GATE_INSTRUCTION = (
    "Is abstract [1] relevant to the claim? Abstracts [2] and [3] are shown for context only. "
    "Reply with exactly one line: ANSWER: yes or ANSWER: no."
)
_ANSWER_LINE = re.compile(r"^\s*ANSWER:\s*(yes|no)\b", re.I | re.M)


@dataclass(frozen=True)
class GateRecord:
    keep_bm25: bool  # True = the gate said "yes, rank 1 is relevant"
    status: str  # yes | no | unparseable | model_error
    usage: Usage | None
    model: str | None
    latency_s: float
    error: str | None = None
    error_kind: str | None = None  # exception class name, e.g. ModelRefused, ModelOutputTruncated
    text: str = ""  # the model's visible answer, for audit
    reasoning_summary: str | None = None  # provider summary of hidden thinking (not evidence of why)
    stop_reason: str | None = None  # the provider's own stop reason


def gate_messages(query: str, texts: Sequence[str]) -> list[Message]:
    """Only the query and the top-3 texts: never ids or labels."""
    if len(texts) != GATE_TOP:
        raise ValueError(f"the gate shows exactly {GATE_TOP} results, got {len(texts)}")
    user = f"Claim: {query}\n\nAbstracts:\n\n{numbered(texts, GATE_MAX_DOC_CHARS)}\n\n{GATE_INSTRUCTION}"
    return [Message("system", GATE_SYSTEM), Message("user", user)]


def parse_gate(text: str) -> str:
    """yes | no | unparseable (the last ANSWER line counts)."""
    answers = _ANSWER_LINE.findall(text)
    return answers[-1].lower() if answers else "unparseable"


def gate(model: ModelClient, query: str, ranking: Sequence[str], corpus: dict[str, str]) -> GateRecord:
    started = time.perf_counter()
    try:
        response = model.complete(gate_messages(query, [corpus[d] for d in ranking[:GATE_TOP]]), [])
    except Exception as exc:
        return GateRecord(False, "model_error", None, None, time.perf_counter() - started,
                          f"{type(exc).__name__}: {exc}", type(exc).__name__)
    status = parse_gate(response.text)
    return GateRecord(status == "yes", status, response.usage, response.model, time.perf_counter() - started,
                      text=response.text, reasoning_summary=response.reasoning_summary,
                      stop_reason=response.stop_reason)


def flip_rate(repeated: dict[str, Sequence[bool]]) -> float:
    """Share of queries whose keep/rerank decision differs across repeated calls."""
    if not repeated or any(len(d) < 2 for d in repeated.values()):
        raise ValueError("every query needs at least two repeated decisions")
    return sum(len(set(d)) > 1 for d in repeated.values()) / len(repeated)


def provenance() -> dict:
    return {"gate_top": GATE_TOP, "gate_max_doc_chars": GATE_MAX_DOC_CHARS,
            "gate_system": GATE_SYSTEM, "gate_instruction": GATE_INSTRUCTION}
