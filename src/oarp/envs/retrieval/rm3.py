"""RM3 pseudo-relevance feedback for BM25: a fixed, model-free "search again".

RM3-style query expansion (after Lavrenko and Croft's relevance model; the
defaults are Anserini's), written in oarp on top of `BM25Index`:

1. Feedback: BM25's top `fb_docs` for the query (documents with score > 0).
2. Relevance model RM1: P(t) proportional to sum over feedback documents D of
   score(D) * tf(t, D) / |D|, with the index's own tokenizer (lower case,
   English stopwords removed, no stemming) and only index-vocabulary terms.
3. Keep the `fb_terms` heaviest terms (ties by term) and renormalise to sum 1.
4. Interpolate with the original query model (token frequency / query length):
   w(t) = lambda * P_query(t) + (1 - lambda) * P_RM1(t), lambda =
   `original_query_weight`.
5. Rank with the weighted BM25 query sum_t w(t) * BM25(t, D).

With lambda = 1 the ranking equals plain BM25 (the weights are the query's
token counts divided by its length, a constant factor). It is RM3-style, not a
reimplementation of Anserini: tokenisation, feedback weighting and scoring
details differ. Uses no labels and no model calls.
"""

from collections import Counter
from dataclasses import asdict, dataclass

from oarp.envs.retrieval.bm25 import BM25Index


@dataclass(frozen=True)
class RM3Config:
    """Fixed in advance (Anserini's defaults), not tuned; recorded with results."""

    fb_docs: int = 10
    fb_terms: int = 10
    original_query_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.fb_docs < 1 or self.fb_terms < 1 or not 0 <= self.original_query_weight <= 1:
            raise ValueError("fb_docs and fb_terms must be >= 1, original_query_weight in [0, 1]")


def expansion_weights(index: BM25Index, corpus: dict[str, str], query: str,
                      config: RM3Config = RM3Config()) -> dict[str, float]:
    """The expanded query as term -> weight (summing to 1); {} if the query has
    no vocabulary term."""
    tokens = index.query_tokens(query)
    if not tokens:
        return {}
    original = {t: c / len(tokens) for t, c in Counter(tokens).items()}
    feedback = [(d, s) for d, s in index.rank(query, config.fb_docs) if s > 0]
    total = sum(s for _, s in feedback)
    rm1: Counter[str] = Counter()
    for doc_id, score in feedback:
        doc_tokens = index.query_tokens(corpus[doc_id])
        for term, count in Counter(doc_tokens).items():
            rm1[term] += (score / total) * count / len(doc_tokens)
    top = sorted(rm1.items(), key=lambda item: (-item[1], item[0]))[: config.fb_terms]
    mass = sum(w for _, w in top)
    expansion = {t: w / mass for t, w in top} if mass > 0 else {}
    lam = config.original_query_weight if expansion else 1.0
    return {t: lam * original.get(t, 0.0) + (1 - lam) * expansion.get(t, 0.0)
            for t in sorted(original.keys() | expansion.keys())}


def rm3_rank(index: BM25Index, corpus: dict[str, str], query: str, config: RM3Config = RM3Config(),
             k: int | None = None) -> list[tuple[str, float]]:
    """(doc id, weighted score) pairs, best first, ties by doc id."""
    return index.rank_weighted(expansion_weights(index, corpus, query, config), k)


def provenance(config: RM3Config) -> dict:
    return {"rm3_config": asdict(config)}
