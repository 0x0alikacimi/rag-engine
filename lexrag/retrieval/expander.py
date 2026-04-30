from __future__ import annotations

from lexrag.ingestion.metadata import MetadataStore
from lexrag.utils.logging import get_logger

logger = get_logger(__name__)


class ParentChunkExpander:
    """Expands child chunks to their parent chunks for richer context."""

    def __init__(self, store: MetadataStore) -> None:
        self._store = store

    def expand(self, chunks: list[dict]) -> list[dict]:
        """For each chunk, try to replace it with its parent chunk.

        If a parent exists, the parent text is used (preserving child metadata).
        Deduplicates by parent_chunk_id so siblings don't add duplicate parents.
        """
        seen_parent_ids: set[str] = set()
        expanded: list[dict] = []

        for chunk in chunks:
            parent_chunk_id = chunk.get("parent_chunk_id")

            if parent_chunk_id and parent_chunk_id not in seen_parent_ids:
                parent = self._store.get_chunk(parent_chunk_id)
                if parent:
                    # Merge: use parent text, keep child scores and metadata
                    merged = dict(parent)
                    merged["retrieval_score"] = chunk.get("retrieval_score", 0.0)
                    merged["rerank_score"] = chunk.get("rerank_score", 0.0)
                    merged["child_chunk_id"] = chunk["chunk_id"]
                    expanded.append(merged)
                    seen_parent_ids.add(parent_chunk_id)
                    continue

            if chunk["chunk_id"] not in {c["chunk_id"] for c in expanded}:
                expanded.append(chunk)

        return expanded
