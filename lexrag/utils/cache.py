from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from config import settings
from lexrag.utils.logging import get_logger

logger = get_logger(__name__)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class EmbeddingCache:
    """Disk-backed cache for embedding vectors."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self._dir = (cache_dir or settings.cache_dir) / "embeddings"
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.npy"

    def get(self, text: str) -> Optional[np.ndarray]:
        if not settings.cache_embeddings:
            return None
        key = _hash_text(text)
        p = self._path(key)
        if p.exists():
            try:
                return np.load(p)
            except Exception as e:
                logger.warning(f"Embedding cache read failed: {e}")
        return None

    def set(self, text: str, embedding: np.ndarray) -> None:
        if not settings.cache_embeddings:
            return
        key = _hash_text(text)
        p = self._path(key)
        try:
            np.save(p, embedding)
        except Exception as e:
            logger.warning(f"Embedding cache write failed: {e}")

    def get_batch(self, texts: list[str]) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for text in texts:
            emb = self.get(text)
            if emb is not None:
                result[text] = emb
        return result

    def set_batch(self, texts: list[str], embeddings: list[np.ndarray]) -> None:
        for text, emb in zip(texts, embeddings):
            self.set(text, emb)


class QueryCache:
    """In-memory + disk TTL cache for query results."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self._dir = (cache_dir or settings.cache_dir) / "queries"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, tuple[Any, float]] = {}

    def _key(self, query: str, top_k: int) -> str:
        raw = f"{query}||{top_k}"
        return _hash_text(raw)

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.pkl"

    def get(self, query: str, top_k: int) -> Optional[Any]:
        if not settings.cache_query_results:
            return None
        key = self._key(query, top_k)
        now = time.time()

        # Memory first
        if key in self._memory:
            val, expires = self._memory[key]
            if now < expires:
                return val
            del self._memory[key]

        # Disk fallback
        p = self._path(key)
        if p.exists():
            try:
                with open(p, "rb") as f:
                    val, expires = pickle.load(f)
                if now < expires:
                    self._memory[key] = (val, expires)
                    return val
                p.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Query cache read failed: {e}")
        return None

    def set(self, query: str, top_k: int, value: Any) -> None:
        if not settings.cache_query_results:
            return
        key = self._key(query, top_k)
        expires = time.time() + settings.cache_query_ttl_seconds
        self._memory[key] = (value, expires)
        p = self._path(key)
        try:
            with open(p, "wb") as f:
                pickle.dump((value, expires), f)
        except Exception as e:
            logger.warning(f"Query cache write failed: {e}")
