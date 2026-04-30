from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from config import settings
from lexrag.evaluation.dataset import EvalSample
from lexrag.utils.logging import get_logger

if TYPE_CHECKING:
    from lexrag.api.server import LexRAGPipeline

logger = get_logger(__name__)


def _text_match(candidate: str, references: list[str]) -> bool:
    """Check if candidate text contains any reference answer (case-insensitive substring)."""
    candidate_lower = candidate.lower()
    for ref in references:
        if ref.lower().strip() in candidate_lower:
            return True
    return False


def _recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    hits = sum(1 for r in relevant if r in top_k)
    return hits / len(relevant)


def _mrr(retrieved: list[str], relevant: list[str]) -> float:
    relevant_set = set(relevant)
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant_set:
            return 1.0 / rank
    return 0.0


class RetrievalEvaluator:
    """Evaluates retrieval quality on a labeled dataset."""

    def __init__(
        self,
        pipeline: "LexRAGPipeline",
        k_values: Optional[list[int]] = None,
    ) -> None:
        self.pipeline = pipeline
        self.k_values = k_values or settings.eval_recall_k_values

    def evaluate(self, samples: list[EvalSample], top_k: int = 10) -> dict:
        """Run retrieval for each sample and compute metrics.

        When relevant_chunk_ids is empty (CUAD style), uses text matching
        against retrieved chunk texts as a proxy for relevance.
        """
        recall_accumulators: dict[int, list[float]] = {k: [] for k in self.k_values}
        mrr_scores: list[float] = []
        num_evaluated = 0

        for sample in samples:
            try:
                result = self.pipeline.query(
                    query=sample.question,
                    top_k=top_k,
                    expand_to_parent=False,
                    rerank=True,
                    generate_answer=False,
                )
            except Exception as e:
                logger.warning(f"Query failed for sample {sample.question_id}: {e}")
                continue

            retrieved_chunks = result.get("chunks", [])
            retrieved_ids = [c["chunk_id"] for c in retrieved_chunks]
            retrieved_texts = [c["text"] for c in retrieved_chunks]

            # Resolve relevant IDs
            if sample.relevant_chunk_ids:
                relevant_ids = sample.relevant_chunk_ids
            else:
                # Text-matching fallback: find chunks that contain the answer span
                relevant_ids = [
                    cid
                    for cid, txt in zip(retrieved_ids, retrieved_texts)
                    if _text_match(txt, sample.relevant_texts)
                ]

            for k in self.k_values:
                recall_accumulators[k].append(
                    _recall_at_k(retrieved_ids, relevant_ids, k)
                )

            mrr_scores.append(_mrr(retrieved_ids, relevant_ids))
            num_evaluated += 1

        if num_evaluated == 0:
            return {"error": "No samples evaluated successfully"}

        metrics: dict = {
            "num_evaluated": num_evaluated,
            "mrr": round(sum(mrr_scores) / len(mrr_scores), 4),
        }
        for k in self.k_values:
            vals = recall_accumulators[k]
            metrics[f"recall@{k}"] = round(sum(vals) / len(vals), 4)

        logger.info(f"Evaluation complete. Metrics: {metrics}")
        return metrics
