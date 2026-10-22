from activity import cursor
from activity.store import Event, EventStore


def fetch_page(store: EventStore, token: str | None, limit: int) -> tuple[list[Event], str | None]:
    after = cursor.decode(token) if token is not None else None
    events = store.page(after, limit)
    next_token = cursor.encode(events[-1]) if len(events) == limit else None
    return events, next_token


def export_all(store: EventStore, page_size: int = 100) -> list[Event]:
    exported: list[Event] = []
    token = None
    seen_tokens: set[str] = set()
    while True:
        events, token = fetch_page(store, token, page_size)
        exported.extend(events)
        if token is None:
            return exported
        if token in seen_tokens:
            raise RuntimeError("page token repeated: paging is not advancing")
        seen_tokens.add(token)
