import pytest

from activity.cursor import InvalidCursor
from activity.export import export_all, fetch_page
from activity.store import Event, EventStore


def ids(events):
    return [e.id for e in events]


def assert_complete_and_oldest_first(store_events, exported):
    assert sorted(ids(exported)) == sorted(ids(store_events)), "each event exactly once"
    times = [e.created_at for e in exported]
    assert times == sorted(times), "oldest first"


def test_empty_store():
    assert export_all(EventStore(), page_size=3) == []


def test_distinct_timestamps_over_several_pages():
    store = EventStore([Event(i, 100 + i, "click") for i in range(1, 6)])
    assert ids(export_all(store, page_size=2)) == [1, 2, 3, 4, 5]


def test_page_size_larger_than_store():
    store = EventStore([Event(1, 10, "view"), Event(2, 20, "view")])
    assert ids(export_all(store, page_size=50)) == [1, 2]


def test_shared_timestamp_across_a_page_boundary():
    events = [
        Event(1, 10, "view"),
        Event(2, 20, "click"),
        Event(3, 20, "click"),
        Event(4, 30, "view"),
    ]
    assert_complete_and_oldest_first(events, export_all(EventStore(events), page_size=2))


def test_first_page_and_last_page():
    store = EventStore([Event(1, 10, "view"), Event(2, 20, "view"), Event(3, 30, "view")])
    first, token = fetch_page(store, None, 2)
    assert ids(first) == [1, 2] and token is not None
    last, token = fetch_page(store, token, 2)
    assert ids(last) == [3] and token is None


def test_invalid_limit():
    with pytest.raises(ValueError):
        fetch_page(EventStore(), None, 0)


def test_garbage_token():
    with pytest.raises(InvalidCursor):
        fetch_page(EventStore(), "not-a-token", 2)
