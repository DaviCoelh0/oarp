import random

import pytest

from activity.export import export_all, fetch_page
from activity.store import Event, EventStore


def ids(events):
    return [e.id for e in events]


def assert_complete_and_oldest_first(store_events, exported):
    assert sorted(ids(exported)) == sorted(ids(store_events)), "each event exactly once"
    times = [e.created_at for e in exported]
    assert times == sorted(times), "oldest first"


@pytest.mark.parametrize("page_size", [1, 2])
def test_three_way_tie_across_two_boundaries(page_size):
    events = [Event(1, 10, "a"), Event(2, 10, "b"), Event(3, 10, "c"), Event(4, 20, "d")]
    assert_complete_and_oldest_first(events, export_all(EventStore(events), page_size=page_size))


def test_ids_need_not_follow_time():
    events = [Event(50, 10, "late id"), Event(1, 20, "a"), Event(2, 20, "b"), Event(3, 30, "c")]
    exported = export_all(EventStore(events), page_size=2)
    assert_complete_and_oldest_first(events, exported)
    assert ids(exported)[0] == 50 and ids(exported)[-1] == 3


def test_exactly_full_last_page():
    store = EventStore([Event(i, 10 * i, "x") for i in range(1, 5)])
    first, token = fetch_page(store, None, 2)
    second, token = fetch_page(store, token, 2)
    assert ids(first + second) == [1, 2, 3, 4]
    rest, token = fetch_page(store, token, 2) if token is not None else ([], None)
    assert rest == [] and token is None


def test_exporting_twice_gives_the_same_order():
    events = [Event(i, 5, "same time") for i in (4, 2, 7, 1)]
    store = EventStore(events)
    first = export_all(store, page_size=3)
    assert_complete_and_oldest_first(events, first)
    assert export_all(store, page_size=3) == first
    assert len(store) == 4


def test_every_page_size_gives_the_same_complete_order():
    rng = random.Random(20261022)
    event_ids = rng.sample(range(1, 1000), 40)
    events = [Event(i, rng.randint(1, 6), "e") for i in event_ids]  # many shared timestamps
    reference = export_all(EventStore(events), page_size=1)
    assert_complete_and_oldest_first(events, reference)
    for page_size in range(2, 9):
        assert export_all(EventStore(events), page_size=page_size) == reference, page_size


def test_invalid_limit_with_a_token():
    store = EventStore([Event(1, 10, "a"), Event(2, 20, "b")])
    _, token = fetch_page(store, None, 1)
    with pytest.raises(ValueError):
        fetch_page(store, token, 0)
