import copy


def merge(base: dict, override: dict) -> dict:
    """Combine `override` into `base`: dicts key by key, other values replaced.

    Values taken from `override` are copied, so the result never shares
    mutable objects with `override`.
    """
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base
