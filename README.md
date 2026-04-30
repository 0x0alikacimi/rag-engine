# LexRAG — Legal Document RAG System

A fully local, production-grade Retrieval-Augmented Generation system for legal documents.

## Architecture

```text
Documents (PDF/DOCX/TXT)
│
▼
DocumentLoader → LegalChunker (structure-aware)
│                │
│          parent/child chunks
▼
MetadataStore (SQLite)
│
▼
Embedder (BGE) ──────────► FAISSIndex (HNSW)
BM25Index ◄──────────────────────┘
│
▼
HybridRetriever (RRF fusion)
│
▼
CrossEncoderReranker
│
▼
ParentChunkExpander
│
▼
ContextSelector (token budget)
│
▼
PromptBuilder → LocalLLM (llama.cpp)
│
▼
Answer + Citations

```

## Installation

### 1. Clone & create environment

```bash
git clone <repo>
cd lexrag
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU acceleration (optional):
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --upgrade
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env — set LLM_MODEL_PATH to your GGUF model
```

Download a GGUF model (e.g. Mistral-7B-Instruct, Llama-3-8B-Instruct):
```bash
# Example: download a quantized model
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
     -O /path/to/model.gguf
```

## Usage

### Ingest documents

Place your legal documents (PDF, DOCX, TXT) into `data/documents/`, then:

```bash
python scripts/index.py
```

Or ingest a specific directory:

```bash
python scripts/index.py --docs-dir /path/to/contracts/
```

Force rebuild indices:

```bash
python scripts/index.py --rebuild
```

### Start the API server

```bash
python main.py
```

The API will be available at `http://localhost:8000`.

Interactive docs: `http://localhost:8000/docs`

### Example queries

**Retrieve + generate answer:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the termination conditions in this agreement?",
    "top_k": 5,
    "generate_answer": true,
    "rerank": true,
    "expand_to_parent": true
  }'
```

**Retrieval only (no LLM):**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "indemnification obligations", "generate_answer": false}'
```

**Ingest a document via API:**
```bash
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@/path/to/contract.pdf"
```

**Health check:**
```bash
curl http://localhost:8000/health
```

### Run evaluation

Prepare a dataset file (see `data/eval/` for format), then:

```bash
python scripts/evaluate.py \
  --dataset data/eval/my_cuad_dataset.json \
  --top-k 10 \
  --max-samples 200 \
  --output results.json
```

**Expected dataset format (flat):**
```json
[
  {
    "id": "q1",
    "question": "What is the governing law?",
    "answer_texts": ["This Agreement shall be governed by the laws of Delaware"],
    "document": "contract_001.pdf"
  }
]
```

**CUAD SQuAD format is also supported** (the standard CUAD JSON).

## Configuration

All settings live in `config.py` and can be overridden via `.env`:

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL_PATH` | None | Path to GGUF model file |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `CHUNK_SIZE` | 512 | Child chunk character size |
| `PARENT_CHUNK_SIZE` | 2048 | Parent chunk character size |
| `TOP_K_RETRIEVAL` | 20 | Candidates before reranking |
| `TOP_K_RERANK` | 5 | Final results after reranking |
| `MAX_CONTEXT_TOKENS` | 2048 | Token budget for LLM context |
| `API_PORT` | 8000 | API server port |

## Retrieval Strategy

1. **Structure-aware chunking** — Splits on legal section headings (ARTICLE, SECTION, numbered clauses) before token-level chunking. Creates parent (2k chars) + child (512 chars) pairs.

2. **Hybrid retrieval** — Dense (BGE embeddings + FAISS HNSW) and sparse (BM25) retrieval are run in parallel, then fused with RRF (Reciprocal Rank Fusion).

3. **Cross-encoder reranking** — The fused candidates are reranked using a ms-marco cross-encoder for precision.

4. **Parent-child expansion** — Retrieved child chunks are expanded to their parent chunks before generation, providing richer context while keeping retrieval granular.

## Performance Notes

- Supports ~100k chunks on CPU with FAISS HNSW indexing
- Embedding: ~3k chunks/sec on CPU with bge-small
- Reranking adds ~100-300ms per query
- LLM inference speed depends on model size and quantization
