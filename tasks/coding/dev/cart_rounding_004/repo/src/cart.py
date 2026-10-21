from decimal import Decimal

from money import round_cents


class Cart:
    def __init__(self) -> None:
        self._lines: list[tuple[str, Decimal, int]] = []

    def add(self, name: str, unit_price: str, quantity: int = 1) -> None:
        if quantity < 1:
            raise ValueError("quantity must be >= 1")
        self._lines.append((name, Decimal(unit_price), quantity))

    def subtotal(self) -> Decimal:
        """Sum of unit price times quantity over all lines, rounded to cents."""
        return round_cents(sum((price * qty for _, price, qty in self._lines), Decimal("0")))

    def total(self, discount_percent: int = 0) -> Decimal:
        """Subtotal minus a percentage discount, rounded to cents."""
        if not 0 <= discount_percent <= 100:
            raise ValueError("discount_percent must be between 0 and 100")
        return round_cents(self.subtotal() * (100 - discount_percent) / 100)
