from __future__ import annotations

from typing import Optional

from config import settings
from lexrag.utils.logging import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Cross-encoder reranker using sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str = settings.reranker_model,
        device: str = settings.reranker_device,
        batch_size: int = settings.reranker_batch_size,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        logger.info(f"Loading cross-encoder reranker: {self.model_name}")
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self.model_name, device=self.device)
        logger.info("Reranker loaded.")

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = settings.top_k_rerank,
    ) -> list[dict]:
        """Rerank a list of chunk dicts (must have 'text' key).

        Returns top_k chunks sorted by cross-encoder score, with 'rerank_score' added.
        """
        if not chunks:
            return []

        self._load_model()

        pairs = [(query, c["text"]) for c in chunks]
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
