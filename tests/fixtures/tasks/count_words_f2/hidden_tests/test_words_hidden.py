from words import count_words


def test_empty_text():
    assert count_words("") == 0


def test_tabs_and_newlines():
    assert count_words("a\tb\nc") == 3, "HIDDEN_SENTINEL_7f3a: tabs and newlines separate words"
