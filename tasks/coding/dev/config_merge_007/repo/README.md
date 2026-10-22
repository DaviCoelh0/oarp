# appconfig

Application configuration: built-in defaults plus per-deployment overrides.

## Loading

`load_config(overrides=None)` returns a configuration dict built from the
defaults in `defaults.py` and the given overrides. Configuration values are
JSON-like: dicts with string keys, lists, strings, numbers, booleans and `None`.

- Where the default and the override are both dicts, they are combined key by
  key, at any depth.
- Any other override value (a list, string, number, boolean or `None`)
  replaces the default value. Lists are replaced, not extended.
- Keys that appear only in the overrides are added.

Every call returns a configuration of its own. Changing a returned
configuration, at any depth, never affects the defaults, configurations
returned by other calls, or the overrides object that was passed in. Changing
the overrides object after the call never affects the returned
configuration. `load_config` does not modify its argument.

## Modes

`enable_debug(config)` switches the given configuration to debug mode, in
place: log level `DEBUG`, an extra `file` log handler, and `127.0.0.1` added
to the allowed hosts.
