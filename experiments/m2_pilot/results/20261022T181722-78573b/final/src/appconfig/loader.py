from appconfig.defaults import DEFAULTS
from appconfig.merge import merge


def load_config(overrides: dict | None = None) -> dict:
    # merge() deep-copies its base, so the shared DEFAULTS are never touched.
    return merge(DEFAULTS, overrides or {})
