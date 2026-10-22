from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    id: int
    created_at: int  # seconds since the epoch
    kind: str


class EventStore:
    """In-memory event storage."""

    def __init__(self, events: list[Event] | None = None) -> None:
        self._events: list[Event] = []
        for event in events or []:
            self.add(event)

    def add(self, event: Event) -> None:
        if any(existing.id == event.id for existing in self._events):
            raise ValueError(f"duplicate event id {event.id}")
        self._events.append(event)

    def __len__(self) -> int:
        return len(self._events)

    def page(self, after: tuple[int, int] | None, limit: int) -> list[Event]:
        """Up to `limit` events in feed order, starting strictly after the
        position `after` = (created_at, id) (None = from the start).

        Feed order is (created_at, id), which is total because ids are unique.
        """
        if limit < 1:
            raise ValueError("limit must be >= 1")
        ordered = sorted(self._events, key=lambda e: (e.created_at, e.id))
        if after is not None:
            ordered = [e for e in ordered if (e.created_at, e.id) > after]
        return ordered[:limit]
