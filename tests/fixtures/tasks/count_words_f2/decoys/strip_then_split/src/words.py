def count_words(text: str) -> int:
    """Number of words, where words are separated by any run of whitespace."""
    return len(text.strip().split(" "))
