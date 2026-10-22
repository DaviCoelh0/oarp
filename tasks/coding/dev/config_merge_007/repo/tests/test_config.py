from appconfig.loader import load_config
from appconfig.modes import enable_debug


def test_defaults():
    config = load_config()
    assert config["database"]["port"] == 5432
    assert config["logging"] == {"level": "INFO", "handlers": ["console"]}


def test_nested_override_keeps_sibling_values():
    config = load_config({"database": {"port": 6000}})
    assert config["database"]["port"] == 6000
    assert config["database"]["host"] == "localhost"
    assert config["database"]["options"] == {"timeout_s": 5, "retries": 3}


def test_debug_mode_changes_only_its_own_config():
    debug = load_config()
    enable_debug(debug)
    assert debug["logging"] == {"level": "DEBUG", "handlers": ["console", "file"]}
    assert debug["allowed_hosts"] == ["localhost", "127.0.0.1"]

    later = load_config()
    assert later["logging"] == {"level": "INFO", "handlers": ["console"]}
    assert later["allowed_hosts"] == ["localhost"]


def test_override_does_not_leak_into_later_loads():
    load_config({"database": {"port": 6000, "options": {"retries": 10}}})
    later = load_config()
    assert later["database"]["port"] == 5432
    assert later["database"]["options"]["retries"] == 3
