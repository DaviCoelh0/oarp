from appconfig.defaults import DEFAULTS
from appconfig.merge import merge


def load_config(overrides: dict | None = None) -> dict:
    # merge() copies both inputs, so the defaults and overrides are never
    # shared with the returned configuration.
    return merge(DEFAULTS, overrides or {})
