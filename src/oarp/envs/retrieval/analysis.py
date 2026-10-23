"""Label-reading analysis helpers: oracle reranker, oracle gate, gate quality and
threshold selection. ANALYSIS ONLY: these read relevance labels, so no strategy
may use them (a test checks that rerank.py, gate.py and strategies.py never
import this module). Use dev labels only; the held-out split stays unloaded.
"""

import math
import statistics
from collections.abc import Mapping, Sequence

from oarp.envs.retrieval.metrics import relevant


def oracle_rerank(ranking: Sequence[str], judged: dict[str, int], k: int) -> list[str]:
    """A perfect reranker of the top k: relevant documents first (higher grade
    first, then BM25 order), then the rest of the top k, then ranks below k."""
    rel = relevant(judged)
    top, rest = list(ranking[:k]), list(ranking[k:])
    hits = sorted((d for d in top if d in rel), key=lambda d: -rel[d])  # stable: BM25 order within a grade
    return hits + [d for d in top if d not in rel] + rest


def rank1_relevant(ranking: Sequence[str], judged: dict[str, int]) -> bool:
    return bool(ranking) and judged.get(ranking[0], 0) > 0


def oracle_gate(bm25: Mapping[str, Sequence[str]], qrels: Mapping[str, dict[str, int]]) -> dict[str, bool]:
    """True = rerank, exactly when BM25's rank 1 is not relevant."""
    return {qid: not rank1_relevant(ranking, qrels[qid]) for qid, ranking in bm25.items()}


def gate_precision_recall(keep_bm25: Mapping[str, bool], qrels: Mapping[str, dict[str, int]],
                          bm25: Mapping[str, Sequence[str]]) -> dict[str, float | None]:
    """Positive class: the gate says "rank 1 is relevant" (keep BM25).

    precision = P(rank 1 relevant | gate says yes); recall = P(gate says yes |
    rank 1 relevant). None where the denominator is empty.
    """
    truth = {qid: rank1_relevant(bm25[qid], qrels[qid]) for qid in keep_bm25}
    tp = sum(keep_bm25[q] and truth[q] for q in keep_bm25)
    said_yes = sum(keep_bm25.values())
    actually = sum(truth.values())
    return {"precision": tp / said_yes if said_yes else None, "recall": tp / actually if actually else None,
            "said_yes": said_yes, "rank1_relevant": actually, "queries": len(keep_bm25)}


def mixed_mean(rerank: Mapping[str, bool], without: Mapping[str, float], with_: Mapping[str, float]) -> float:
    """Mean per-query metric when each query takes the reranked value iff rerank[qid]."""
    return statistics.fmean(with_[q] if rerank[q] else without[q] for q in rerank)


def choose_tau(gaps: Mapping[str, float], without: Mapping[str, float], with_: Mapping[str, float]) -> float:
    """The tau with the best mean metric on the given (tune) queries; ties go to
    the smaller tau (fewer reranks). Candidates: -inf, every gap, +inf."""
    candidates = [-math.inf, *sorted(set(gaps.values())), math.inf]
    return max(candidates, key=lambda t: (mixed_mean({q: g < t for q, g in gaps.items()}, without, with_), -t))
