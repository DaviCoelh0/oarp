import pytest

from appconfig.loader import load_config
from appconfig.modes import enable_debug

PRISTINE = {
    "database": {"host": "localhost", "port": 5432, "options": {"timeout_s": 5, "retries": 3}},
    "logging": {"level": "INFO", "handlers": ["console"]},
    "features": {"beta": False},
    "allowed_hosts": ["localhost"],
}


def scribble(value):
    """Change every dict and list inside `value`, at every depth."""
    if isinstance(value, dict):
        for child in list(value.values()):
            scribble(child)
        value["scribbled"] = True
    elif isinstance(value, list):
        value.append("scribbled")


def test_deep_changes_to_a_config_never_reach_the_next_one():
    first = load_config()
    scribble(first)
    assert load_config() == PRISTINE


def test_two_configs_are_independent():
    a, b = load_config(), load_config()
    a["features"]["beta"] = True
    a["database"]["options"]["timeout_s"] = 99
    assert b["features"]["beta"] is False and b["database"]["options"]["timeout_s"] == 5


def test_three_levels_deep_override():
    config = load_config({"database": {"options": {"timeout_s": 30}}})
    assert config["database"] == {"host": "localhost", "port": 5432, "options": {"timeout_s": 30, "retries": 3}}


@pytest.mark.parametrize(("override", "expected"), [
    ({"allowed_hosts": ["example.org"]}, ["example.org"]),
    ({"allowed_hosts": []}, []),
])
def test_lists_are_replaced(override, expected):
    assert load_config(override)["allowed_hosts"] == expected


def test_non_dict_override_replaces_a_section_and_back():
    assert load_config({"features": None})["features"] is None
    assert load_config({"database": {"port": {"primary": 1, "replica": 2}}})["database"]["port"] == {
        "primary": 1, "replica": 2}


def test_new_keys_are_added():
    assert load_config({"cache": {"size": 10}})["cache"] == {"size": 10}


def test_overrides_are_not_modified():
    overrides = {"database": {"options": {"retries": 7}}, "allowed_hosts": ["a"]}
    config = load_config(overrides)
    enable_debug(config)
    scribble(config)
    assert overrides == {"database": {"options": {"retries": 7}}, "allowed_hosts": ["a"]}


def test_later_changes_to_overrides_do_not_reach_the_config():
    overrides = {"allowed_hosts": ["a"], "cache": {"size": 10}}
    config = load_config(overrides)
    overrides["allowed_hosts"].append("b")
    overrides["cache"]["size"] = 0
    assert config["allowed_hosts"] == ["a"] and config["cache"] == {"size": 10}


def test_debug_then_override_then_plain_load():
    enable_debug(load_config({"logging": {"level": "WARNING"}}))
    load_config({"database": {"options": {"retries": 0}}})
    assert load_config() == PRISTINE
