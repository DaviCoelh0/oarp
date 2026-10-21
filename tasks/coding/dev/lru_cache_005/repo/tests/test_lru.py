import pytest

from lru import LRUCache


def test_put_and_get():
    cache = LRUCache(2)
    cache.put("a", 1)
    assert cache.get("a") == 1


def test_missing_key_returns_default():
    assert LRUCache(1).get("nope", "fallback") == "fallback"


def test_least_recently_put_is_evicted():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert cache.get("a") is None and cache.get("c") == 3


def test_get_counts_as_a_use():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")
    cache.put("c", 3)
    assert cache.get("a") == 1
    assert cache.get("b") is None


def test_updating_a_key_counts_as_a_use():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("a", 10)
    cache.put("c", 3)
    assert cache.get("a") == 10 and cache.get("b") is None


def test_eviction_order_after_several_gets():
    cache = LRUCache(3)
    for key in "abc":
        cache.put(key, key.upper())
    cache.get("a")
    cache.get("b")
    cache.put("d", "D")
    assert cache.get("c") is None
    assert [cache.get(k) for k in "abd"] == ["A", "B", "D"]


def test_size_never_exceeds_capacity():
    cache = LRUCache(2)
    for i in range(10):
        cache.put(str(i), i)
    assert len(cache) == 2


def test_invalid_capacity():
    with pytest.raises(ValueError):
        LRUCache(0)
