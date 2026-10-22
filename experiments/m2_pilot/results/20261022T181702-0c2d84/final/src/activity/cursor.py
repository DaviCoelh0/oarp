"""Page tokens: opaque, URL-safe strings that mark a position in the feed."""

import base64
import binascii
import json

from activity.store import Event, feed_key


class InvalidCursor(ValueError):
    pass


def encode(last: Event) -> str:
    """Token for the position just after `last`."""
    created_at, event_id = feed_key(last)
    payload = json.dumps({"t": created_at, "i": event_id})
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")


def decode(token: str) -> tuple[int, int]:
    try:
        payload = json.loads(base64.urlsafe_b64decode(token.encode("ascii")))
        return (int(payload["t"]), int(payload["i"]))
    except (binascii.Error, UnicodeError, ValueError, KeyError, TypeError) as exc:
        raise InvalidCursor(f"not a valid page token: {token!r}") from exc
