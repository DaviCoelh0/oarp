"""Label-free strategy pieces: BM25 confidence signals, the S2 rule gate, and the
offline combiner that turns S1 rankings plus gate decisions into final rankings.

Strategies (all end in a full ranking scored at the same cut-offs):
- S0: BM25 order; S1: the reranked ranking for every query;
- S2: rerank when the relative gap (s1 - s2) / s1 between BM25's top two
  scores is below tau (s1 <= 0 counts as gap 0, i.e. rerank);
- S2b: rerank when the model gate does not say "rank 1 is relevant".

Because a gate only chooses between two rankings that already exist, S2 and
S2b are scored offline from S1's outputs: the only extra paid cost of S2b is
its gate calls. Nothing here reads relevance labels (see `analysis.py`).
"""

import bisect
import math
from collections.abc import Mapping, Sequence

from oarp.core.pricing import estimate_cost


def bm25_signals(scored: Sequence[tuple[str, float]]) -> dict[str, float]:
    """Confidence signals from a BM25 ranking with scores, best first."""
    s1 = scored[0][1] if scored else 0.0
    s2 = scored[1][1] if len(scored) > 1 else 0.0
    return {"relative_gap": 0.0 if s1 <= 0 else (s1 - s2) / s1, "raw_gap": s1 - s2, "top_score": s1}


def rule_gate(gaps: Mapping[str, float], tau: float) -> dict[str, bool]:
    """S2: True = rerank (gap below tau)."""
    return {qid: gap < tau for qid, gap in gaps.items()}


def tau_for_rate(gaps: Sequence[float], rate: float) -> float:
    """The threshold whose rerank share (gap < tau) is closest to `rate`.

    Candidates are -inf, every observed gap and +inf; ties go to the smaller
    tau (fewer reranks). With tied gaps the achieved rate can differ from the
    target; callers report the achieved rate.
    """
    if not gaps:
        raise ValueError("no gaps")
    ordered = sorted(gaps)
    candidates = [-math.inf, *sorted(set(ordered)), math.inf]
    return min(candidates, key=lambda t: (abs(bisect.bisect_left(ordered, t) / len(ordered) - rate), t))


def combine(bm25: Mapping[str, Sequence[str]], reranked: Mapping[str, Sequence[str]],
            rerank: Mapping[str, bool]) -> dict[str, list[str]]:
    """Final rankings: the reranked one where `rerank` is True, BM25's otherwise."""
    if not (set(bm25) == set(reranked) == set(rerank)):
        raise ValueError("BM25 rankings, reranked rankings and decisions must cover the same queries")
    return {qid: list(reranked[qid] if rerank[qid] else bm25[qid]) for qid in bm25}


def usage_totals(records: Sequence[object], model: str) -> dict[str, int | float | None]:
    """Calls, tokens and estimated $ over rerank or gate records (anything with
    `.usage`). Records without usage (model errors) count as calls with
    unknown tokens; the estimate is then None, never guessed."""
    usages = [r.usage for r in records]  # type: ignore[attr-defined]
    known = [u for u in usages if u is not None and u.input_tokens is not None and u.output_tokens is not None]
    tokens_in = sum(u.input_tokens for u in known)
    tokens_out = sum(u.output_tokens for u in known)
    costs = [estimate_cost(model, u) for u in known]
    complete = len(known) == len(usages) and all(c is not None for c in costs)
    return {"calls": len(usages), "input_tokens": tokens_in, "output_tokens": tokens_out,
            "calls_without_usage": len(usages) - len(known),
            "estimated_usd": sum(costs) if complete else None}  # type: ignore[arg-type]


def gate_decisions(keep_bm25: Mapping[str, bool]) -> dict[str, bool]:
    """S2b: a model gate's keep-BM25 answers as rerank decisions."""
    return {qid: not keep for qid, keep in keep_bm25.items()}
