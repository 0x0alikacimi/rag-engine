#!/usr/bin/env python3
"""Run retrieval evaluation on a CUAD-style dataset.

Usage:
    python scripts/evaluate.py --dataset data/eval/my_dataset.json
    python scripts/evaluate.py --dataset data/eval/my_dataset.json --top-k 10 --max-samples 100
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from lexrag.api.server import LexRAGPipeline
from lexrag.evaluation.dataset import CUADDatasetLoader
from lexrag.evaluation.metrics import RetrievalEvaluator
from lexrag.utils.logging import get_logger

logger = get_logger("scripts.evaluate")


def main():
    parser = argparse.ArgumentParser(description="LexRAG retrieval evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to CUAD-style JSON dataset",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve per query",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples evaluated",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write JSON results",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    # Load pipeline
    logger.info("Loading pipeline and indices...")
    pipeline = LexRAGPipeline()
    pipeline.try_load_indices()

    if pipeline.faiss_index.size == 0:
        logger.error("Index is empty. Run scripts/index.py first.")
        sys.exit(1)

    # Load dataset
    loader = CUADDatasetLoader(str(dataset_path))
    samples = loader.load()
    if args.max_samples:
        samples = samples[: args.max_samples]

    logger.info(f"Evaluating {len(samples)} samples with top_k={args.top_k}...")

    # Evaluate
    evaluator = RetrievalEvaluator(pipeline)
    metrics = evaluator.evaluate(samples, top_k=args.top_k)

    print("\n=== EVALUATION RESULTS ===")
    for key, val in metrics.items():
        print(f"  {key:20s}: {val}")
    print()

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(
            json.dumps(
                {
                    "dataset": str(dataset_path),
                    "num_samples": len(samples),
                    "top_k": args.top_k,
                    "metrics": metrics,
                },
                indent=2,
            )
        )
        logger.info(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
