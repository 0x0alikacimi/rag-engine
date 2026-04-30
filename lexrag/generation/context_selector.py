from __future__ import annotations

from config import settings
from lexrag.utils.tokens import token_count
from lexrag.utils.logging import get_logger

logger = get_logger(__name__)


class ContextSelector:
    """Selects chunks that fit within the token budget."""

    def __init__(self, max_tokens: int = settings.max_context_tokens) -> None:
        self.max_tokens = max_tokens

    def select(self, chunks: list[dict]) -> list[dict]:
        """Greedily select chunks in score order until budget is exhausted."""
        selected: list[dict] = []
        used_tokens = 0

        for chunk in chunks:
            text = chunk.get("text", "")
            cost = token_count(text)

            if used_tokens + cost > self.max_tokens:
                # Try to squeeze a truncated version
                remaining = self.max_tokens - used_tokens
                if remaining > 50:
                    truncated = text[: int(remaining / settings.tokens_per_char_estimate)]
                    chunk = dict(chunk)
                    chunk["text"] = truncated
                    selected.append(chunk)
                break

            selected.append(chunk)
            used_tokens += cost

        logger.debug(
            f"Context selection: {len(selected)}/{len(chunks)} chunks, "
            f"~{used_tokens} tokens used of {self.max_tokens} budget"
        )
        return selected
