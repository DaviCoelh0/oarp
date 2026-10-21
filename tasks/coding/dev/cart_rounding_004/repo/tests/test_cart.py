from decimal import Decimal

import pytest

from cart import Cart


def test_empty_cart_is_free():
    assert Cart().total() == Decimal("0.00")


def test_quantities_are_summed():
    cart = Cart()
    cart.add("pen", "1.50", 2)
    cart.add("pad", "3.25")
    assert cart.total() == Decimal("6.25")


def test_exact_discount():
    cart = Cart()
    cart.add("book", "20.00")
    assert cart.total(10) == Decimal("18.00")


def test_discount_rounds_half_up():
    cart = Cart()
    cart.add("item", "10.05")
    assert cart.total(50) == Decimal("5.03")  # 5.025 -> 5.03


def test_subtotal_rounds_half_up():
    cart = Cart()
    cart.add("bolt", "0.125", 3)
    assert cart.subtotal() == Decimal("0.38")  # 0.375 -> 0.38


def test_invalid_discount():
    with pytest.raises(ValueError):
        Cart().total(150)
