def enable_debug(config: dict) -> None:
    config["logging"]["level"] = "DEBUG"
    config["logging"]["handlers"].append("file")
    config["allowed_hosts"].append("127.0.0.1")
