from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).parent.resolve()


class Settings(BaseSettings):
	model_config = SettingsConfigDict(
		env_file=".env",
		env_file_encoding="utf-8",
		case_sensitive=False,
		extra="ignore",
	)

	# ── Paths ──────────────────────────────────────────────────────────────
	data_dir: Path = BASE_DIR / "data"
	documents_dir: Path = BASE_DIR / "data" / "documents"
	indices_dir: Path = BASE_DIR / "data" / "indices"
	cache_dir: Path = BASE_DIR / "data" / "cache"
	eval_dir: Path = BASE_DIR / "data" / "eval"
	db_path: Path = BASE_DIR / "data" / "lexrag.db"
	log_dir: Path = BASE_DIR / "logs"

	# ── Embedding model ────────────────────────────────────────────────────
	embedding_model: str = "BAAI/bge-small-en-v1.5"
	embedding_dim: int = 384
	embedding_batch_size: int = 32
	embedding_device: str = "cpu"

	# ── FAISS ──────────────────────────────────────────────────────────────
	faiss_index_type: str = "hnsw"          # "hnsw" | "flat"
	faiss_hnsw_m: int = 32
	faiss_hnsw_ef_construction: int = 200
	faiss_hnsw_ef_search: int = 128
	faiss_index_file: str = "faiss.index"
	faiss_metadata_file: str = "faiss_meta.json"

	# ── BM25 ───────────────────────────────────────────────────────────────
	bm25_k1: float = 1.5
	bm25_b: float = 0.75
	bm25_index_file: str = "bm25.pkl"

	# ── Hybrid / RRF ───────────────────────────────────────────────────────
	rrf_k: int = 60
	hybrid_dense_weight: float = 0.6
	hybrid_sparse_weight: float = 0.4
	top_k_retrieval: int = 20
	top_k_rerank: int = 5

	# ── Cross-encoder reranker ─────────────────────────────────────────────
	reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
	reranker_device: str = "cpu"
	reranker_batch_size: int = 16

	# ── Chunking ───────────────────────────────────────────────────────────
	chunk_size: int = 512
	chunk_overlap: int = 64
	min_chunk_size: int = 64
	parent_chunk_size: int = 2048

	# ── LLM (llama-cpp-python) ────────────────────────────────────────────
	llm_model_path: Optional[str] = None    # required at runtime
	llm_n_ctx: int = 4096
	llm_n_threads: int = 4
	llm_n_gpu_layers: int = 0               # 0 = CPU only
	llm_temperature: float = 0.1
	llm_max_tokens: int = 1024
	llm_top_p: float = 0.95
	llm_repeat_penalty: float = 1.1

	# ── Context / token budget ────────────────────────────────────────────
	max_context_tokens: int = 2048
	tokens_per_char_estimate: float = 0.25  # rough char → token ratio

	# ── API ────────────────────────────────────────────────────────────────
	api_host: str = "0.0.0.0"
	api_port: int = 8000
	api_workers: int = 1
	api_reload: bool = False
	api_log_level: str = "info"

	# ── Cache ──────────────────────────────────────────────────────────────
	cache_embeddings: bool = True
	cache_query_results: bool = True
	cache_query_ttl_seconds: int = 3600

	# ── Evaluation ────────────────────────────────────────────────────────
	eval_recall_k_values: list[int] = [1, 3, 5, 10]

	def create_dirs(self) -> None:
		for d in [
			self.data_dir,
			self.documents_dir,
			self.indices_dir,
			self.cache_dir,
			self.eval_dir,
			self.log_dir,
		]:
			d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.create_dirs()
