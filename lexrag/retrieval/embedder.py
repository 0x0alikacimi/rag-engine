from __future__ import annotations

from typing import Optional
import numpy as np

from config import settings
from lexrag.utils.logging import get_logger
from lexrag.utils.cache import EmbeddingCache

logger = get_logger(__name__)


class Embedder:
    """Wraps sentence-transformers for batch embedding with caching."""

    def __init__(
        self,
        model_name: str = settings.embedding_model,
        device: str = settings.embedding_device,
        batch_size: int = settings.embedding_batch_size,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._cache = EmbeddingCache()

    def _load_model(self) -> None:
        if self._model is not None:
            return
        logger.info(f"Loading embedding model: {self.model_name}")
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name, device=self.device)
        logger.info("Embedding model loaded.")

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns (N, dim) float32 array."""
        self._load_model()

        # Separate cached and uncached
        cached_map = self._cache.get_batch(texts)
        missing_texts = [t for t in texts if t not in cached_map]

        if missing_texts:
            logger.debug(f"Embedding {len(missing_texts)} uncached texts")
            embeddings = self._model.encode(
                missing_texts,
                batch_size=self.batch_size,
                show_progress_bar=len(missing_texts) > 100,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            embeddings = embeddings.astype(np.float32)
            self._cache.set_batch(missing_texts, [embeddings[i] for i in range(len(missing_texts))])
            for text, emb in zip(missing_texts, embeddings):
                cached_map[text] = emb

        result = np.stack([cached_map[t] for t in texts], axis=0)
        return result.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns (dim,) float32 array."""
        result = self.embed_texts([query])
        return result[0]

    @property
    def dim(self) -> int:
        return settings.embedding_dim
