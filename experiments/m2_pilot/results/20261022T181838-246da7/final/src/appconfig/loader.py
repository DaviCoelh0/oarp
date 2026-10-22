from appconfig.defaults import DEFAULTS
from appconfig.merge import merge


def load_config(overrides: dict | None = None) -> dict:
    # merge() deep-copies its inputs, so the returned config never shares
    # mutable state with DEFAULTS or with the caller's overrides.
    return merge(DEFAULTS, overrides or {})
