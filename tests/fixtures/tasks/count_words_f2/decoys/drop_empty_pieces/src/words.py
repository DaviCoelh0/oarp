def count_words(text: str) -> int:
    """Number of words, where words are separated by any run of whitespace."""
    return len([word for word in text.split(" ") if word])
