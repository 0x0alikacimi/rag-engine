from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException

from lexrag.api.models import (
    ChunkResult,
    Citation,
    QueryRequest,
    QueryResponse,
)
from lexrag.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/query", tags=["query"])


def _get_pipeline():
    from lexrag.api.server import get_pipeline
    return get_pipeline()


@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Retrieve relevant chunks and optionally generate an answer."""
    pipeline = _get_pipeline()

    if pipeline.faiss_index.size == 0:
        raise HTTPException(503, "Index is empty. Ingest documents first.")

    t0 = time.perf_counter()

    try:
        result = pipeline.query(
            query=request.query,
            top_k=request.top_k,
            expand_to_parent=request.expand_to_parent,
            rerank=request.rerank,
            generate_answer=request.generate_answer,
        )
    except Exception as e:
        logger.exception("Query pipeline error")
        raise HTTPException(500, f"Query error: {e}")

    retrieval_ms = result.get("retrieval_ms", 0.0)
    generation_ms = result.get("generation_ms")

    chunks = [
        ChunkResult(
            chunk_id=c["chunk_id"],
            doc_id=c["doc_id"],
            doc_name=c.get("metadata", {}).get("source", c.get("doc_id", "")),
            section_heading=c.get("section_heading", ""),
            text=c["text"],
            retrieval_score=c.get("retrieval_score", 0.0),
            rerank_score=c.get("rerank_score"),
        )
        for c in result.get("chunks", [])
    ]

    citations = [
        Citation(**cit) for cit in result.get("citations", [])
    ]

    return QueryResponse(
        query=request.query,
        answer=result.get("answer"),
        citations=citations,
        chunks=chunks,
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
    )
