from words import count_words


def test_single_spaces():
    assert count_words("a b c") == 3


def test_repeated_spaces():
    assert count_words("a  b") == 2


def test_leading_space():
    assert count_words(" a b") == 2
