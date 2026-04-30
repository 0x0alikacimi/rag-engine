#!/bin/zsh

set -e

# Top-level files
touch main.py config.py requirements.txt README.md .env.example

# Scripts
mkdir -p scripts
touch scripts/index.py scripts/evaluate.py

# Package structure
mkdir -p lexrag/{ingestion,retrieval,generation,api/routes,evaluation,utils}

# __init__ files
touch lexrag/__init__.py
touch lexrag/ingestion/__init__.py
touch lexrag/retrieval/__init__.py
touch lexrag/generation/__init__.py
touch lexrag/api/__init__.py
touch lexrag/api/routes/__init__.py
touch lexrag/evaluation/__init__.py
touch lexrag/utils/__init__.py

# Ingestion
touch lexrag/ingestion/loaders.py
touch lexrag/ingestion/chunker.py
touch lexrag/ingestion/metadata.py

# Retrieval
touch lexrag/retrieval/embedder.py
touch lexrag/retrieval/faiss_index.py
touch lexrag/retrieval/bm25_index.py
touch lexrag/retrieval/hybrid.py
touch lexrag/retrieval/reranker.py
touch lexrag/retrieval/expander.py

# Generation
touch lexrag/generation/prompt_builder.py
touch lexrag/generation/context_selector.py
touch lexrag/generation/llm.py

# API
touch lexrag/api/server.py
touch lexrag/api/models.py
touch lexrag/api/routes/query.py
touch lexrag/api/routes/ingest.py
touch lexrag/api/routes/evaluate.py

# Evaluation
touch lexrag/evaluation/dataset.py
touch lexrag/evaluation/metrics.py

# Utils
touch lexrag/utils/logging.py
touch lexrag/utils/cache.py
touch lexrag/utils/tokens.py

# Data directories
mkdir -p data/{documents,indices,cache,eval}

echo " ----------------------- Project structure created successfully! ----------------------- "
