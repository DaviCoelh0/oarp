def add_tag(item: str, tags: list[str] | None = None) -> list[str]:
    """Return a list of tags with `item` added (no duplicates, order kept).

    If `tags` is given, the result starts from it, but the caller's list must
    not be modified. Without `tags`, start from an empty list.
    """
    result = list(tags) if tags is not None else []
    if item not in result:
        result.append(item)
    return result
