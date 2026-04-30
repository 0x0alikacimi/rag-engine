from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lexrag.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvalSample:
    """One QA sample for evaluation."""
    question_id: str
    question: str
    relevant_chunk_ids: list[str]       # Ground-truth chunk IDs (may be empty)
    relevant_texts: list[str]           # Ground-truth answer spans
    document_name: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class CUADDatasetLoader:
    """Loads CUAD-style evaluation datasets.

    Expected JSON format:
    {
      "data": [
        {
          "title": "Contract Title",
          "paragraphs": [
            {
              "context": "...",
              "qas": [
                {
                  "id": "unique_id",
                  "question": "...",
                  "answers": [{"text": "...", "answer_start": 0}],
                  "is_impossible": false
                }
              ]
            }
          ]
        }
      ]
    }

    Also accepts a simpler flat format:
    [
      {
        "id": "...",
        "question": "...",
        "answer_texts": ["..."],
        "document": "contract.pdf"
      }
    ]
    """

    def __init__(self, dataset_path: str) -> None:
        self.path = Path(dataset_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    def load(self) -> list[EvalSample]:
        with open(self.path) as f:
            data = json.load(f)

        if isinstance(data, list):
            return self._load_flat(data)
        elif isinstance(data, dict) and "data" in data:
            return self._load_cuad(data)
        else:
            raise ValueError("Unsupported dataset format.")

    def _load_cuad(self, data: dict) -> list[EvalSample]:
        samples: list[EvalSample] = []
        for article in data["data"]:
            title = article.get("title", "")
            for paragraph in article.get("paragraphs", []):
                for qa in paragraph.get("qas", []):
                    if qa.get("is_impossible", False):
                        continue
                    answers = [a["text"] for a in qa.get("answers", []) if a.get("text")]
                    samples.append(
                        EvalSample(
                            question_id=qa["id"],
                            question=qa["question"],
                            relevant_chunk_ids=[],   # filled by text matching at eval time
                            relevant_texts=answers,
                            document_name=title,
                        )
                    )
        logger.info(f"Loaded {len(samples)} CUAD samples from {self.path.name}")
        return samples

    def _load_flat(self, data: list) -> list[EvalSample]:
        samples: list[EvalSample] = []
        for item in data:
            samples.append(
                EvalSample(
                    question_id=str(item.get("id", "")),
                    question=item["question"],
                    relevant_chunk_ids=item.get("relevant_chunk_ids", []),
                    relevant_texts=item.get("answer_texts", []),
                    document_name=item.get("document"),
                )
            )
        logger.info(f"Loaded {len(samples)} flat samples from {self.path.name}")
        return samples
