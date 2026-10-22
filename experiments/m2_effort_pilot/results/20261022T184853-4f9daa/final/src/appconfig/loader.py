from appconfig.defaults import DEFAULTS
from appconfig.merge import merge


def load_config(overrides: dict | None = None) -> dict:
    return merge(DEFAULTS, overrides or {})
