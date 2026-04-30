from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from lexrag.api.models import HealthResponse
from lexrag.api.routes import ingest, query, evaluate
from lexrag.ingestion import DocumentLoader, LegalChunker, MetadataStore
from lexrag.retrieval import (
    Embedder,
    FAISSIndex,
    BM25Index,
    HybridRetriever,
    CrossEncoderReranker,
    ParentChunkExpander,
)
from lexrag.generation import PromptBuilder, ContextSelector, LocalLLM
from lexrag.utils.cache import QueryCache
from lexrag.utils.logging import get_logger

logger = get_logger(__name__)


class LexRAGPipeline:
    """End-to-end pipeline wiring all components together."""

    def __init__(self) -> None:
        self.store = MetadataStore()
        self.loader = DocumentLoader()
        self.chunker = LegalChunker()
        self.embedder = Embedder()
        self.faiss_index = FAISSIndex()
        self.bm25_index = BM25Index()
        self.reranker = CrossEncoderReranker()
        self.expander = ParentChunkExpander(self.store)
        self.prompt_builder = PromptBuilder()
        self.context_selector = ContextSelector()
        self.llm = LocalLLM()
        self.query_cache = QueryCache()
        self._indices_loaded = False

    # ── Index management ───────────────────────────────────────────────────

    def try_load_indices(self) -> None:
        if self._indices_loaded:
            return
        try:
            if self.faiss_index.is_built():
                self.faiss_index.load()
            if self.bm25_index.is_built():
                self.bm25_index.load()
            self._indices_loaded = True
            logger.info(
                f"Indices loaded: FAISS={self.faiss_index.size}, BM25={self.bm25_index.size}"
            )
        except Exception as e:
            logger.warning(f"Could not load indices: {e}")

    def rebuild_indices(self) -> None:
        """Rebuild both indices from the metadata store."""
        import numpy as np

        logger.info("Rebuilding indices from stored chunks...")
        chunks = self.store.get_all_child_chunks()
        if not chunks:
            logger.warning("No child chunks found; index not built.")
            return

        texts = [c["text"] for c in chunks]
        chunk_ids = [c["chunk_id"] for c in chunks]

        # Embeddings
        embeddings = self.embedder.embed_texts(texts)

        # FAISS
        self.faiss_index.build(embeddings, chunk_ids)
        self.faiss_index.save()

        # BM25
        self.bm25_index.build(texts, chunk_ids)
        self.bm25_index.save()

        self._indices_loaded = True
        logger.info("Indices rebuilt and saved.")

    # ── Ingestion ──────────────────────────────────────────────────────────

    def ingest_document(self, file_path: str) -> dict:
        """Ingest a single document and update indices incrementally."""
        import numpy as np

        doc = self.loader.load(file_path)

        if self.store.document_exists(doc.doc_id):
            logger.info(f"Document already ingested: {doc.file_name}")
            chunks = self.store.get_chunks_by_doc(doc.doc_id)
            child_chunks = [c for c in chunks if c.get("parent_chunk_id")]
            return {"doc_id": doc.doc_id, "chunk_count": len(child_chunks)}

        parents, children = self.chunker.chunk(doc)

        self.store.upsert_document(
            doc_id=doc.doc_id,
            file_path=doc.file_path,
            file_name=doc.file_name,
            file_type=doc.file_type,
            char_count=len(doc.raw_text),
            chunk_count=len(children),
            metadata=doc.metadata,
        )
        self.store.insert_chunks(parents)
        self.store.insert_chunks(children)

        # Incremental index update
        if children:
            texts = [c.text for c in children]
            chunk_ids = [c.chunk_id for c in children]
            embeddings = self.embedder.embed_texts(texts)
            self.rebuild_indices()  # Rebuild for simplicity (fast for <100k)

        logger.info(f"Ingested: {doc.file_name} → {len(children)} child chunks")
        return {"doc_id": doc.doc_id, "chunk_count": len(children)}

    # ── Query ──────────────────────────────────────────────────────────────

    def query(
        self,
        query: str,
        top_k: int = settings.top_k_rerank,
        expand_to_parent: bool = True,
        rerank: bool = True,
        generate_answer: bool = True,
    ) -> dict:
        # Cache check
        cached = self.query_cache.get(query, top_k)
        if cached is not None:
            logger.debug("Query cache hit")
            return cached

        t0 = time.perf_counter()

        # Embed query
        query_embedding = self.embedder.embed_query(query)

        # Hybrid retrieval
        retriever = HybridRetriever(self.faiss_index, self.bm25_index)
        fused_results = retriever.retrieve(
            query, query_embedding, top_k=settings.top_k_retrieval
        )

        # Fetch chunk metadata
        retrieved_chunks: list[dict] = []
        for chunk_id, score in fused_results:
            chunk = self.store.get_chunk(chunk_id)
            if chunk:
                chunk["retrieval_score"] = score
                retrieved_chunks.append(chunk)

        # Rerank
        if rerank and retrieved_chunks:
            retrieved_chunks = self.reranker.rerank(
                query, retrieved_chunks, top_k=settings.top_k_retrieval
            )

        # Parent expansion
        if expand_to_parent:
            retrieved_chunks = self.expander.expand(retrieved_chunks)

        # Trim to top_k after expansion
        retrieved_chunks = retrieved_chunks[:top_k]

        retrieval_ms = (time.perf_counter() - t0) * 1000

        result: dict = {
            "chunks": retrieved_chunks,
            "retrieval_ms": retrieval_ms,
        }

        # Generation
        if generate_answer and self.llm.is_available:
            t1 = time.perf_counter()
            context = self.context_selector.select(retrieved_chunks)
            prompt, citations = self.prompt_builder.build(query, context)
            answer = self.llm.generate(prompt)
            result["answer"] = answer
            result["citations"] = citations
            result["generation_ms"] = (time.perf_counter() - t1) * 1000
        else:
            result["answer"] = None
            result["citations"] = []

        self.query_cache.set(query, top_k, result)
        return result


# Global pipeline singleton
_pipeline: Optional[LexRAGPipeline] = None


def get_pipeline() -> LexRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = LexRAGPipeline()
        _pipeline.try_load_indices()
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("LexRAG API starting up...")
    get_pipeline()
    yield
    logger.info("LexRAG API shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="LexRAG",
        description="Legal Document Retrieval-Augmented Generation System",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(ingest.router)
    app.include_router(query.router)
    app.include_router(evaluate.router)

    @app.get("/health", response_model=HealthResponse)
    async def health():
        pipeline = get_pipeline()
        docs = pipeline.store.list_documents()
        return HealthResponse(
            status="ok",
            index_size=pipeline.faiss_index.size,
            document_count=len(docs),
            llm_available=pipeline.llm.is_available,
        )

    return app
