"""Small statistics for paired retrieval comparisons, written by hand (numpy only).

- paired_bootstrap_ci: mean of per-query differences with a percentile
  bootstrap CI, resampling queries with replacement. Queries are the unit of
  resampling; repeated calls of one query are not independent samples.
- kendall_tau: Kendall's tau-a between two orderings of the same items
  (no ties possible: each is a ranking).
- spearman: Spearman's rho between two sequences of distinct values.
"""

from collections.abc import Sequence
from itertools import combinations

import numpy as np


def paired_bootstrap_ci(differences: Sequence[float], *, resamples: int = 10_000, seed: int = 0,
                        level: float = 0.95) -> dict[str, float]:
    if not differences:
        raise ValueError("no differences")
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    alpha = (1 - level) / 2
    return {"mean": float(values.mean()), "low": float(np.quantile(means, alpha)),
            "high": float(np.quantile(means, 1 - alpha)), "n": len(values), "resamples": resamples,
            "seed": seed, "level": level}


def kendall_tau(first: Sequence[str], second: Sequence[str]) -> float:
    if sorted(first) != sorted(second) or len(set(first)) != len(first):
        raise ValueError("both orderings must contain the same distinct items")
    if len(first) < 2:
        raise ValueError("need at least two items")
    rank = {item: i for i, item in enumerate(second)}
    pairs = list(combinations(range(len(first)), 2))
    concordant = sum(rank[first[i]] < rank[first[j]] for i, j in pairs)
    return (2 * concordant - len(pairs)) / len(pairs)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("need two sequences of equal length >= 2")
    if len(set(xs)) != len(xs) or len(set(ys)) != len(ys):
        raise ValueError("values must be distinct (no ties)")
    rx = {x: r for r, x in enumerate(sorted(xs))}
    ry = {y: r for r, y in enumerate(sorted(ys))}
    n = len(xs)
    d2 = sum((rx[x] - ry[y]) ** 2 for x, y in zip(xs, ys))
    return 1 - 6 * d2 / (n * (n * n - 1))
