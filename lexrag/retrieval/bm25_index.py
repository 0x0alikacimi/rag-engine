from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Optional

from config import settings
from lexrag.utils.logging import get_logger

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


class BM25Index:
    """BM25 lexical search index using rank_bm25."""

    def __init__(self, index_dir: Optional[Path] = None) -> None:
        self._index_dir = index_dir or settings.indices_dir
        self._bm25 = None
        self._chunk_ids: list[str] = []

    def build(self, texts: list[str], chunk_ids: list[str]) -> None:
        from rank_bm25 import BM25Okapi

        assert len(texts) == len(chunk_ids)
        logger.info(f"Building BM25 index with {len(texts)} documents")

        tokenized = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(
            tokenized,
            k1=settings.bm25_k1,
            b=settings.bm25_b,
        )
        self._chunk_ids = chunk_ids
        logger.info("BM25 index built.")

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            raise RuntimeError("BM25 index not loaded.")

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        # Get top_k indices
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[tuple[str, float]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                results.append((self._chunk_ids[idx], score))
        return results

    def save(self) -> None:
        path = self._index_dir / settings.bm25_index_file
        with open(path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "chunk_ids": self._chunk_ids}, f)
        logger.info(f"BM25 index saved to {path}")

    def load(self) -> None:
        path = self._index_dir / settings.bm25_index_file
        if not path.exists():
            raise FileNotFoundError(f"BM25 index not found at {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._chunk_ids = data["chunk_ids"]
        logger.info(f"BM25 index loaded: {len(self._chunk_ids)} documents")

    def is_built(self) -> bool:
        return (self._index_dir / settings.bm25_index_file).exists()

    @property
    def size(self) -> int:
        return len(self._chunk_ids)
