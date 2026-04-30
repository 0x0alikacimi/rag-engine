from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from config import settings
from lexrag.api.models import IngestResponse
from lexrag.ingestion import DocumentLoader, LegalChunker, MetadataStore
from lexrag.retrieval import Embedder, FAISSIndex, BM25Index
from lexrag.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingest"])


def _get_pipeline():
    from lexrag.api.server import get_pipeline
    return get_pipeline()


@router.post("/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """Upload and ingest a legal document (PDF, DOCX, TXT)."""
    pipeline = _get_pipeline()

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".docx", ".doc", ".txt", ".text"}:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    # Save upload to documents dir
    dest = settings.documents_dir / file.filename
    content = await file.read()
    dest.write_bytes(content)

    try:
        result = pipeline.ingest_document(str(dest))
    except Exception as e:
        logger.exception(f"Ingest failed for {file.filename}")
        raise HTTPException(500, f"Ingestion error: {e}")

    return IngestResponse(
        doc_id=result["doc_id"],
        file_name=file.filename,
        chunk_count=result["chunk_count"],
        status="success",
        message=f"Ingested {result['chunk_count']} chunks from {file.filename}",
    )


@router.post("/path", response_model=IngestResponse)
async def ingest_path(file_path: str):
    """Ingest a document by local file path."""
    pipeline = _get_pipeline()
    path = Path(file_path)

    if not path.exists():
        raise HTTPException(404, f"File not found: {file_path}")

    try:
        result = pipeline.ingest_document(str(path))
    except Exception as e:
        logger.exception(f"Ingest failed for {file_path}")
        raise HTTPException(500, f"Ingestion error: {e}")

    return IngestResponse(
        doc_id=result["doc_id"],
        file_name=path.name,
        chunk_count=result["chunk_count"],
        status="success",
        message=f"Ingested {result['chunk_count']} chunks from {path.name}",
    )
