def enable_debug(config: dict) -> None:
    logging = config["logging"]
    config["logging"] = {**logging, "level": "DEBUG", "handlers": logging["handlers"] + ["file"]}
    config["allowed_hosts"] = config["allowed_hosts"] + ["127.0.0.1"]
