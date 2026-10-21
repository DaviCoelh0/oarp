from decimal import Decimal


def round_cents(amount: Decimal) -> Decimal:
    """Round a money amount to whole cents, with halves rounded up.

    Examples: 2.675 -> 2.68, 2.674 -> 2.67, 0.375 -> 0.38.
    """
    return Decimal(int(amount * 100)) / 100
