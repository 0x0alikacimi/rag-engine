from __future__ import annotations

from fastapi import APIRouter, HTTPException

from lexrag.api.models import EvaluateRequest, EvaluateResponse
from lexrag.evaluation.dataset import CUADDatasetLoader
from lexrag.evaluation.metrics import RetrievalEvaluator
from lexrag.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/evaluate", tags=["evaluate"])


def _get_pipeline():
    from lexrag.api.server import get_pipeline
    return get_pipeline()


@router.post("", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest):
    """Run retrieval evaluation on a CUAD-style dataset."""
    pipeline = _get_pipeline()

    try:
        loader = CUADDatasetLoader(request.dataset_path)
        samples = loader.load()
    except Exception as e:
        raise HTTPException(400, f"Dataset load error: {e}")

    if request.max_samples:
        samples = samples[: request.max_samples]

    evaluator = RetrievalEvaluator(pipeline, k_values=None)
    try:
        metrics = evaluator.evaluate(samples, top_k=request.top_k)
    except Exception as e:
        logger.exception("Evaluation error")
        raise HTTPException(500, f"Evaluation error: {e}")

    return EvaluateResponse(
        dataset_path=request.dataset_path,
        num_samples=len(samples),
        metrics=metrics,
    )
