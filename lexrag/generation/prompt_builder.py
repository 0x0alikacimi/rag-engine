from __future__ import annotations

from config import settings
from lexrag.utils.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are an expert legal assistant. Your task is to answer the user's question based solely on the provided legal document excerpts.

Rules:
- Answer only from the provided context.
- If the context does not contain sufficient information, say "The provided documents do not contain enough information to answer this question."
- Always cite the source document and section when making claims.
- Be precise and use appropriate legal terminology.
- Do not speculate or add information not present in the context."""

_CONTEXT_TEMPLATE = """[Document: {doc_name} | Section: {section} | Chunk: {chunk_id}]
{text}"""

_PROMPT_TEMPLATE = """{system}

=== LEGAL DOCUMENT EXCERPTS ===
{context_block}

=== QUESTION ===
{question}

=== ANSWER ==="""


class PromptBuilder:
    """Assembles the final prompt for LLM generation with citations."""

    def build(
        self,
        query: str,
        context_chunks: list[dict],
        system_prompt: str = _SYSTEM_PROMPT,
    ) -> tuple[str, list[dict]]:
        """Build the full prompt and return (prompt_str, citation_list).

        Citation list: list of dicts with doc_name, section, chunk_id.
        """
        context_parts: list[str] = []
        citations: list[dict] = []

        for i, chunk in enumerate(context_chunks, start=1):
            doc_name = chunk.get("metadata", {}).get("source", chunk.get("doc_id", "unknown"))
            section = chunk.get("section_heading", "Unknown Section")
            chunk_id = chunk.get("chunk_id", "")[:8]
            text = chunk.get("text", "").strip()

            context_parts.append(
                _CONTEXT_TEMPLATE.format(
                    doc_name=doc_name,
                    section=section,
                    chunk_id=chunk_id,
                )
                + "\n"
                + text
            )
            citations.append(
                {
                    "index": i,
                    "doc_name": doc_name,
                    "section": section,
                    "chunk_id": chunk.get("chunk_id", ""),
                    "score": chunk.get("rerank_score", chunk.get("retrieval_score", 0.0)),
                }
            )

        context_block = "\n\n---\n\n".join(context_parts)

        prompt = _PROMPT_TEMPLATE.format(
            system=system_prompt,
            context_block=context_block,
            question=query,
        )

        return prompt, citations
