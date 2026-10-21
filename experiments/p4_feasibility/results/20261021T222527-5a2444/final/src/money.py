from decimal import ROUND_HALF_UP, Decimal


def round_cents(amount: Decimal) -> Decimal:
    """Round a money amount to whole cents, with halves rounded up.

    Examples: 2.675 -> 2.68, 2.674 -> 2.67, 0.375 -> 0.38.
    """
    return amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
