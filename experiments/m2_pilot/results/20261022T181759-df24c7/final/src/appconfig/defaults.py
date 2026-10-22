DEFAULTS = {
    "database": {
        "host": "localhost",
        "port": 5432,
        "options": {"timeout_s": 5, "retries": 3},
    },
    "logging": {"level": "INFO", "handlers": ["console"]},
    "features": {"beta": False},
    "allowed_hosts": ["localhost"],
}
