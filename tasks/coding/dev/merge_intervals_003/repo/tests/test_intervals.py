from intervals import merge_intervals


def test_empty():
    assert merge_intervals([]) == []


def test_disjoint_intervals_are_sorted():
    assert merge_intervals([(5, 6), (1, 2)]) == [(1, 2), (5, 6)]


def test_overlapping_intervals_merge():
    assert merge_intervals([(1, 4), (2, 3), (3, 6)]) == [(1, 6)]


def test_contained_interval_is_absorbed():
    assert merge_intervals([(1, 10), (2, 3)]) == [(1, 10)]


def test_touching_intervals_merge():
    assert merge_intervals([(1, 2), (2, 3)]) == [(1, 3)]


def test_chain_of_touching_intervals():
    assert merge_intervals([(3, 4), (1, 2), (2, 3)]) == [(1, 4)]
