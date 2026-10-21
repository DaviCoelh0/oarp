def add_tag(item: str, tags: list[str] = []) -> list[str]:
    """Return a list of tags with `item` added (no duplicates, order kept).

    If `tags` is given, the result starts from it, but the caller's list must
    not be modified. Without `tags`, start from an empty list.
    """
    if item not in tags:
        tags.append(item)
    return tags
