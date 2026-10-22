import copy


def merge(base: dict, override: dict) -> dict:
    """Return a new dict combining `override` into `base`.

    Dicts are combined key by key; other values in `override` replace those in
    `base`. Neither argument is modified, and the result shares no mutable
    structure with either input.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
