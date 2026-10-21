import pytest

from stats import moving_average


def test_basic_window():
    assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]


def test_window_of_one_returns_values():
    assert moving_average([4, 5, 6], 1) == [4, 5, 6]


def test_window_equal_to_length():
    assert moving_average([2, 4, 6], 3) == [4]


def test_window_longer_than_values():
    assert moving_average([1, 2], 3) == []


def test_invalid_window():
    with pytest.raises(ValueError):
        moving_average([1, 2, 3], 0)
