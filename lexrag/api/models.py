from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Ingest ────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    doc_id: str
    file_name: str
    chunk_count: int
    status: str
    message: str


# ── Query ─────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    expand_to_parent: bool = Field(default=True)
    generate_answer: bool = Field(default=True)
    rerank: bool = Field(default=True)


class ChunkResult(BaseModel):
    chunk_id: str
    doc_id: str
    doc_name: str
    section_heading: str
    text: str
    retrieval_score: float
    rerank_score: Optional[float] = None


class Citation(BaseModel):
    index: int
    doc_name: str
    section: str
    chunk_id: str
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: Optional[str] = None
    citations: list[Citation] = Field(default_factory=list)
    chunks: list[ChunkResult] = Field(default_factory=list)
    retrieval_ms: float
    generation_ms: Optional[float] = None


# ── Evaluate ───────────────────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    dataset_path: str
    top_k: int = Field(default=10, ge=1, le=50)
    max_samples: Optional[int] = None


class EvaluateResponse(BaseModel):
    dataset_path: str
    num_samples: int
    metrics: dict[str, Any]


# ── System ────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    index_size: int
    document_count: int
    llm_available: bool
