from tags import add_tag


def test_single_call_without_tags():
    assert add_tag("a") == ["a"]


def test_independent_calls_do_not_share_state():
    add_tag("x")
    assert add_tag("y") == ["y"]


def test_existing_tags_are_extended():
    assert add_tag("b", ["a"]) == ["a", "b"]


def test_no_duplicates():
    assert add_tag("a", ["a"]) == ["a"]


def test_callers_list_is_not_modified():
    original = ["a"]
    add_tag("b", original)
    assert original == ["a"]
