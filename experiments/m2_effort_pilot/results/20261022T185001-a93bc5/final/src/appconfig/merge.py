import copy


def merge(base: dict, override: dict) -> dict:
    """Combine `override` into a copy of `base`: dicts key by key, other values replaced.

    Neither argument is modified, and the result shares no mutable objects with them.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
