"""Dated price table for ESTIMATED cost. The provider's bill is authoritative.

Prices are USD per token, from the official pricing page (checked 2026-10-19).
Claude Haiku 5.5 has two rate cards chosen by prompt length (all input tokens
of a request, including cache reads/writes; oarp does not use caching yet).
"""

from oarp.core.types import Usage

PRICES_AS_OF = "2026-10-19"

# model -> list of (max prompt tokens, input $/token, output $/token), ascending
PRICES: dict[str, list[tuple[float, float, float]]] = {
    "claude-haiku-5-5": [(100_000, 0.10 / 1e6, 0.50 / 1e6), (float("inf"), 0.50 / 1e6, 2.50 / 1e6)],
    "claude-sonnet-5-5": [(float("inf"), 2.00 / 1e6, 10.00 / 1e6)],
}


def estimate_cost(model: str | None, usage: Usage | None) -> float | None:
    """None when the model is unknown or usage is missing (never guessed as 0)."""
    if model not in PRICES or usage is None or usage.input_tokens is None or usage.output_tokens is None:
        return None
    for max_prompt, price_in, price_out in PRICES[model]:
        if usage.input_tokens <= max_prompt:
            return usage.input_tokens * price_in + usage.output_tokens * price_out
    return None
