from __future__ import annotations

from typing import Optional

import numpy as np

from config import settings
from lexrag.retrieval.faiss_index import FAISSIndex
from lexrag.retrieval.bm25_index import BM25Index
from lexrag.utils.logging import get_logger

logger = get_logger(__name__)


def _reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = settings.rrf_k,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion."""
    rrf_scores: dict[str, float] = {}

    for ranked in ranked_lists:
        for rank, (chunk_id, _) in enumerate(ranked, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items


class HybridRetriever:
    """Combines FAISS dense retrieval and BM25 sparse retrieval via RRF."""

    def __init__(
        self,
        faiss_index: FAISSIndex,
        bm25_index: BM25Index,
        dense_weight: float = settings.hybrid_dense_weight,
        sparse_weight: float = settings.hybrid_sparse_weight,
    ) -> None:
        self.faiss = faiss_index
        self.bm25 = bm25_index
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = settings.top_k_retrieval,
    ) -> list[tuple[str, float]]:
        """Return top_k (chunk_id, rrf_score) pairs."""

        # Dense retrieval – fetch extra candidates before fusion
        fetch_k = min(top_k * 2, self.faiss.size)
        dense_results = self.faiss.search(query_embedding, fetch_k)

        # Sparse retrieval
        sparse_results = self.bm25.search(query, fetch_k)

        if not dense_results and not sparse_results:
            return []

        # RRF fusion
        fused = _reciprocal_rank_fusion([dense_results, sparse_results], k=settings.rrf_k)

        logger.debug(
            f"Hybrid retrieval: {len(dense_results)} dense, "
            f"{len(sparse_results)} sparse → {len(fused)} fused"
        )

        return fused[:top_k]
