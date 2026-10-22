import copy

from appconfig.defaults import DEFAULTS
from appconfig.merge import merge


def load_config(overrides: dict | None = None) -> dict:
    config = dict(DEFAULTS)
    return copy.deepcopy(merge(config, overrides or {}))
