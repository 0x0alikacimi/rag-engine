from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from config import settings
from lexrag.utils.logging import get_logger

logger = get_logger(__name__)


class FAISSIndex:
    """FAISS vector index supporting HNSW and Flat (L2) modes."""

    def __init__(
        self,
        dim: int = settings.embedding_dim,
        index_type: str = settings.faiss_index_type,
        index_dir: Optional[Path] = None,
    ) -> None:
        self.dim = dim
        self.index_type = index_type
        self._index_dir = index_dir or settings.indices_dir
        self._index = None
        self._id_map: list[str] = []   # position → chunk_id

    # ── Build / save / load ────────────────────────────────────────────────

    def build(self, embeddings: np.ndarray, chunk_ids: list[str]) -> None:
        """Build the index from an (N, dim) float32 array."""
        import faiss

        assert embeddings.shape[1] == self.dim, "Embedding dimension mismatch"
        assert embeddings.shape[0] == len(chunk_ids), "Mismatch between embeddings and ids"

        logger.info(f"Building FAISS {self.index_type.upper()} index with {len(chunk_ids)} vectors")

        if self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self.dim, settings.faiss_hnsw_m)
            index.hnsw.efConstruction = settings.faiss_hnsw_ef_construction
            index.hnsw.efSearch = settings.faiss_hnsw_ef_search
        else:
            index = faiss.IndexFlatIP(self.dim)  # Inner product (cosine if normalized)

        index.add(embeddings)
        self._index = index
        self._id_map = chunk_ids
        logger.info("FAISS index built.")

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Return list of (chunk_id, score) sorted by descending score."""
        if self._index is None:
            raise RuntimeError("FAISS index not loaded. Call build() or load() first.")

        if query_vec.ndim == 1:
            query_vec = query_vec[np.newaxis, :]

        scores, indices = self._index.search(query_vec, top_k)
        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            results.append((self._id_map[idx], float(score)))
        return results

    def save(self) -> None:
        import faiss
        if self._index is None:
            raise RuntimeError("No index to save.")

        index_path = self._index_dir / settings.faiss_index_file
        meta_path = self._index_dir / settings.faiss_metadata_file

        faiss.write_index(self._index, str(index_path))
        with open(meta_path, "w") as f:
            json.dump({"id_map": self._id_map, "dim": self.dim, "type": self.index_type}, f)

        logger.info(f"FAISS index saved to {index_path}")

    def load(self) -> None:
        import faiss
        index_path = self._index_dir / settings.faiss_index_file
        meta_path = self._index_dir / settings.faiss_metadata_file

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"FAISS index files not found in {self._index_dir}. "
                "Run `scripts/index.py` first."
            )

        self._index = faiss.read_index(str(index_path))
        with open(meta_path) as f:
            meta = json.load(f)
        self._id_map = meta["id_map"]
        self.dim = meta["dim"]
        self.index_type = meta["type"]

        # Restore ef_search for HNSW
        if self.index_type == "hnsw":
            self._index.hnsw.efSearch = settings.faiss_hnsw_ef_search

        logger.info(f"FAISS index loaded: {len(self._id_map)} vectors")

    def is_built(self) -> bool:
        return (self._index_dir / settings.faiss_index_file).exists()

    @property
    def size(self) -> int:
        return len(self._id_map)
