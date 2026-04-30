from __future__ import annotations

from config import settings


def token_count(text: str) -> int:
    """Rough token count estimate using character ratio."""
    return int(len(text) * settings.tokens_per_char_estimate)


def truncate_to_token_budget(text: str, budget: int) -> str:
    """Truncate text so its estimated token count fits within budget."""
    max_chars = int(budget / settings.tokens_per_char_estimate)
    if len(text) <= max_chars:
        return text
    return text[:max_chars]
