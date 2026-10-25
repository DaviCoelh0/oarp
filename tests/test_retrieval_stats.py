"""Hand-checked statistics for paired retrieval comparisons."""

import pytest

pytest.importorskip("numpy")

from oarp.envs.retrieval.stats import kendall_tau, paired_bootstrap_ci, spearman, stratified_bootstrap_ci  # noqa: E402


def test_kendall_tau_by_hand() -> None:
    assert kendall_tau(["a", "b", "c", "d"], ["a", "b", "c", "d"]) == 1.0
    assert kendall_tau(["a", "b", "c", "d"], ["d", "c", "b", "a"]) == -1.0
    # One swapped adjacent pair out of 6 pairs: (5 - 1) / 6.
    assert kendall_tau(["a", "b", "c", "d"], ["b", "a", "c", "d"]) == pytest.approx(4 / 6)
    with pytest.raises(ValueError):
        kendall_tau(["a", "b"], ["a", "c"])


def test_spearman_by_hand() -> None:
    assert spearman([0, 1, 2], [0, 1, 2]) == 1.0
    assert spearman([0, 1, 2], [2, 1, 0]) == -1.0
    # Ranks (0,1,2,3) vs (1,0,2,3): sum d^2 = 2, rho = 1 - 6*2 / (4*15) = 0.8.
    assert spearman([0, 1, 2, 3], [1, 0, 2, 3]) == pytest.approx(0.8)
    with pytest.raises(ValueError):
        spearman([0, 0], [0, 1])


def test_bootstrap_constant_differences_give_a_degenerate_ci() -> None:
    ci = paired_bootstrap_ci([0.25] * 30)
    assert ci["mean"] == ci["low"] == ci["high"] == pytest.approx(0.25) and ci["n"] == 30


def test_bootstrap_is_seeded_and_covers_the_mean() -> None:
    diffs = [0.0, 0.5, -0.2, 0.3, 0.1, 0.0, 0.4, -0.1]
    first, again = paired_bootstrap_ci(diffs, seed=7), paired_bootstrap_ci(diffs, seed=7)
    assert first == again
    assert first["low"] < first["mean"] < first["high"]
    assert paired_bootstrap_ci(diffs, seed=8) != first
    with pytest.raises(ValueError):
        paired_bootstrap_ci([])


def test_stratified_bootstrap_weights_stratum_means() -> None:
    # Hand example: stratum a always 1, b always 0; population shares 1 : 3.
    ci = stratified_bootstrap_ci({"a": [1.0, 1.0], "b": [0.0, 0.0, 0.0]}, {"a": 1, "b": 3})
    assert ci["mean"] == pytest.approx(0.25) and ci["low"] == pytest.approx(0.25) and ci["high"] == pytest.approx(0.25)
    assert ci["n"] == {"a": 2, "b": 3} and ci["weights"] == {"a": 0.25, "b": 0.75}
    # The weights are population shares, not sample shares: equal samples, unequal weights.
    assert stratified_bootstrap_ci({"a": [1.0], "b": [0.0]}, {"a": 0.156, "b": 0.844})["mean"] == pytest.approx(0.156)


def test_stratified_bootstrap_with_one_stratum_equals_the_paired_bootstrap() -> None:
    values = [0.3, -0.1, 0.0, 0.5, 0.2, -0.4]
    one = stratified_bootstrap_ci({"all": values}, {"all": 1.0}, seed=7)
    paired = paired_bootstrap_ci(values, seed=7)
    assert (one["mean"], one["low"], one["high"]) == pytest.approx((paired["mean"], paired["low"], paired["high"]))


def test_stratified_bootstrap_rejects_missing_weights_or_empty_strata() -> None:
    with pytest.raises(ValueError):
        stratified_bootstrap_ci({"a": [1.0]}, {"b": 1.0})
    with pytest.raises(ValueError):
        stratified_bootstrap_ci({"a": []}, {"a": 1.0})
