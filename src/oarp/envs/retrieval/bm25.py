"""BM25 over an in-memory corpus, via bm25s (optional extra: pip install -e ".[retrieval]").

Every query scores the WHOLE corpus, and the full ranking is ordered by score
(descending), then document id (ascending), before any cut-off: documents
tied at the top-k boundary are therefore chosen deterministically. With a few
thousand documents this costs milliseconds per query.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib import metadata

import bm25s
import numpy as np
from bm25s.tokenization import Tokenizer


@dataclass(frozen=True)
class BM25Config:
    """Everything that determines the ranking; recorded with every result."""

    k1: float = 1.5
    b: float = 0.75
    method: str = "lucene"
    lower: bool = True
    stopwords: str = "english"  # bm25s's built-in English list
    splitter: str = r"(?u)\b\w\w+\b"  # bm25s's default: words of 2+ word characters
    stemmer: None = None  # no stemming in this version


class BM25Index:
    def __init__(self, corpus: dict[str, str], config: BM25Config = BM25Config()) -> None:
        if not corpus:
            raise ValueError("cannot index an empty corpus")
        self.config = config
        self.doc_ids = list(corpus)
        # Position of each document in doc-id order: the secondary sort key.
        self._id_rank = np.empty(len(self.doc_ids), dtype=np.int64)
        self._id_rank[np.argsort(np.array(self.doc_ids, dtype=object), kind="stable")] = np.arange(len(self.doc_ids))
        self._tokenizer = Tokenizer(lower=config.lower, splitter=config.splitter,
                                    stopwords=config.stopwords, stemmer=config.stemmer)
        tokens = self._tokenizer.tokenize([corpus[d] for d in self.doc_ids], return_as="string",
                                          show_progress=False)
        self._bm25 = bm25s.BM25(k1=config.k1, b=config.b, method=config.method)
        self._bm25.index(tokens, show_progress=False)

    def query_tokens(self, text: str) -> list[str]:
        """Query tokens that exist in the index vocabulary (others cannot score)."""
        tokens = self._tokenizer.tokenize([text], update_vocab="never", return_as="string",
                                          show_progress=False)[0]
        return [t for t in tokens if t in self._bm25.vocab_dict and t != ""]

    def scores(self, text: str) -> np.ndarray:
        tokens = self.query_tokens(text)
        if not tokens:
            return np.zeros(len(self.doc_ids), dtype=np.float32)
        return self._bm25.get_scores(tokens)

    def weighted_scores(self, weights: Mapping[str, float]) -> np.ndarray:
        """Sum of weight * BM25 score of each term (bm25s scores are additive over
        query tokens, so weights of 1 per token occurrence equal `scores`). Terms
        outside the vocabulary are ignored."""
        total = np.zeros(len(self.doc_ids), dtype=np.float64)
        for term, weight in weights.items():
            if term in self._bm25.vocab_dict and term != "":
                total += weight * self._bm25.get_scores([term])
        return total

    def rank(self, text: str, k: int | None = None) -> list[tuple[str, float]]:
        """(doc id, score) pairs, best first, ties by doc id; the top k if given."""
        return self._ordered(self.scores(text), k)

    def rank_weighted(self, weights: Mapping[str, float], k: int | None = None) -> list[tuple[str, float]]:
        """As `rank`, for a weighted query (term -> weight)."""
        return self._ordered(self.weighted_scores(weights), k)

    def _ordered(self, scores: np.ndarray, k: int | None) -> list[tuple[str, float]]:
        order = np.lexsort((self._id_rank, -scores))  # last key is primary
        if k is not None:
            order = order[:k]
        return [(self.doc_ids[i], float(scores[i])) for i in order]


def provenance(config: BM25Config) -> dict:
    return {"bm25_config": asdict(config), "bm25s": metadata.version("bm25s"), "numpy": metadata.version("numpy")}
