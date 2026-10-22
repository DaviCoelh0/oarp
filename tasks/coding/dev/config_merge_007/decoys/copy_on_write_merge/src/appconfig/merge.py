def merge(base: dict, override: dict) -> dict:
    """Combine `override` into `base`: dicts key by key, other values replaced."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result
