#!/usr/bin/env python3
"""Build FAISS + BM25 indices from all ingested documents.

Usage:
    python scripts/index.py
    python scripts/index.py --docs-dir /path/to/docs
    python scripts/index.py --rebuild
"""

import argparse
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from lexrag.api.server import LexRAGPipeline
from lexrag.utils.logging import get_logger

logger = get_logger("scripts.index")


def ingest_directory(pipeline: LexRAGPipeline, docs_dir: Path) -> None:
    """Ingest all supported documents in a directory."""
    supported = {".pdf", ".docx", ".doc", ".txt", ".text"}
    files = [f for f in docs_dir.iterdir() if f.suffix.lower() in supported]

    if not files:
        logger.warning(f"No supported files found in {docs_dir}")
        return

    logger.info(f"Found {len(files)} document(s) to ingest")
    for i, fp in enumerate(files, 1):
        logger.info(f"[{i}/{len(files)}] Ingesting: {fp.name}")
        try:
            result = pipeline.ingest_document(str(fp))
            logger.info(f"  → doc_id={result['doc_id']}, chunks={result['chunk_count']}")
        except Exception as e:
            logger.error(f"  ✗ Failed to ingest {fp.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Build LexRAG indices")
    parser.add_argument(
        "--docs-dir",
        type=str,
        default=str(settings.documents_dir),
        help="Directory containing legal documents to ingest",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild indices even if already built",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Skip ingestion; only rebuild indices from stored chunks",
    )
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        logger.error(f"Documents directory not found: {docs_dir}")
        sys.exit(1)

    pipeline = LexRAGPipeline()

    if not args.index_only:
        ingest_directory(pipeline, docs_dir)

    total_chunks = pipeline.store.get_total_chunk_count()
    logger.info(f"Total indexed chunks in store: {total_chunks}")

    if total_chunks == 0:
        logger.error("No chunks in store. Ingest documents first.")
        sys.exit(1)

    if args.rebuild or not pipeline.faiss_index.is_built():
        logger.info("Building indices...")
        pipeline.rebuild_indices()
        logger.info("Index build complete.")
    else:
        logger.info("Indices already exist. Use --rebuild to force rebuild.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
