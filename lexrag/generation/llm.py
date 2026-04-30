from __future__ import annotations

from typing import Optional, Iterator

from config import settings
from lexrag.utils.logging import get_logger

logger = get_logger(__name__)


class LocalLLM:
    """Wrapper around llama-cpp-python for local LLM inference."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = settings.llm_n_ctx,
        n_threads: int = settings.llm_n_threads,
        n_gpu_layers: int = settings.llm_n_gpu_layers,
        temperature: float = settings.llm_temperature,
        max_tokens: int = settings.llm_max_tokens,
    ) -> None:
        self.model_path = model_path or settings.llm_model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm = None

    def _load(self) -> None:
        if self._llm is not None:
            return
        if not self.model_path:
            raise ValueError(
                "LLM model path not set. Set LLM_MODEL_PATH in .env or config."
            )
        logger.info(f"Loading LLM from: {self.model_path}")
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise RuntimeError("llama-cpp-python not installed.") from e

        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )
        logger.info("LLM loaded.")

    def generate(self, prompt: str) -> str:
        """Generate a completion for the given prompt."""
        self._load()

        output = self._llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=settings.llm_top_p,
            repeat_penalty=settings.llm_repeat_penalty,
            stop=["=== QUESTION ===", "=== ANSWER ===", "[Document:"],
            echo=False,
        )
        text: str = output["choices"][0]["text"].strip()
        return text

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """Streaming generation."""
        self._load()

        stream = self._llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=settings.llm_top_p,
            repeat_penalty=settings.llm_repeat_penalty,
            stop=["=== QUESTION ===", "[Document:"],
            echo=False,
            stream=True,
        )
        for chunk in stream:
            delta = chunk["choices"][0]["text"]
            if delta:
                yield delta

    @property
    def is_available(self) -> bool:
        return bool(self.model_path)
