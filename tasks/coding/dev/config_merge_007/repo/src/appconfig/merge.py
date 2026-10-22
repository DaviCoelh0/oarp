def merge(base: dict, override: dict) -> dict:
    """Combine `override` into `base`: dicts key by key, other values replaced."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base
