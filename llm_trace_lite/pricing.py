"""USD per 1M tokens (input, output).

Approximate list prices, copied by hand. They change often and don't include
caching discounts, batch pricing or regional differences. Check the provider's
pricing page before using these for anything that matters.
"""

PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-5-sonnet": (3.00, 15.00),
}


def lookup(model: str) -> tuple[float, float] | None:
    return PRICES.get(model)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    price = lookup(model)
    if price is None:
        return None
    inp, out = price
    return (prompt_tokens * inp + completion_tokens * out) / 1_000_000
