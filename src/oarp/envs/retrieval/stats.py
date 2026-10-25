"""Small statistics for paired retrieval comparisons, written by hand (numpy only).

- paired_bootstrap_ci: mean of per-query differences with a percentile
  bootstrap CI, resampling queries with replacement. Queries are the unit of
  resampling; repeated calls of one query are not independent samples.
- stratified_bootstrap_ci: a weighted mean of per-stratum means (weights =
  population shares) with a percentile CI, resampling queries with
  replacement WITHIN each stratum (a stratified sample's bootstrap).
- kendall_tau: Kendall's tau-a between two orderings of the same items
  (no ties possible: each is a ranking).
- spearman: Spearman's rho between two sequences of distinct values.
"""

from collections.abc import Mapping, Sequence
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


def stratified_bootstrap_ci(strata: Mapping[str, Sequence[float]], weights: Mapping[str, float], *,
                            resamples: int = 10_000, seed: int = 0, level: float = 0.95) -> dict:
    """Estimate sum_s w_s * mean(strata[s]) with weights normalised to sum 1.
    With one stratum this equals `paired_bootstrap_ci` (same draws, same seed)."""
    if set(strata) != set(weights) or not strata or any(len(v) == 0 for v in strata.values()):
        raise ValueError("every stratum needs a weight and at least one value")
    total = sum(weights.values())
    rng = np.random.default_rng(seed)
    estimate = 0.0
    means = np.zeros(resamples)
    for name in sorted(strata):
        values = np.asarray(strata[name], dtype=float)
        w = weights[name] / total
        estimate += w * float(values.mean())
        means += w * values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    alpha = (1 - level) / 2
    return {"mean": estimate, "low": float(np.quantile(means, alpha)), "high": float(np.quantile(means, 1 - alpha)),
            "n": {name: len(v) for name, v in sorted(strata.items())},
            "weights": {name: weights[name] / total for name in sorted(strata)},
            "resamples": resamples, "seed": seed, "level": level}


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
