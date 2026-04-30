from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from config import settings
from lexrag.ingestion.loaders import LoadedDocument
from lexrag.utils.logging import get_logger

logger = get_logger(__name__)


# Legal section heading patterns (covers numbered sections, articles, clauses)
_SECTION_PATTERNS = [
    re.compile(r"^(ARTICLE\s+[IVXLCDM\d]+[\.\:]?\s+.+)$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^(SECTION\s+\d+[\.\:]?\s+.+)$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^(\d+\.\s+[A-Z][A-Z\s]+)$", re.MULTILINE),
    re.compile(r"^(\d+\.\d+[\.\s]+.+)$", re.MULTILINE),
    re.compile(r"^(SCHEDULE\s+[A-Z\d]+[\.\:]?\s*.*)$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^(EXHIBIT\s+[A-Z\d]+[\.\:]?\s*.*)$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^(WHEREAS[,\s].+)$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^(NOW,\s*THEREFORE.+)$", re.MULTILINE | re.IGNORECASE),
]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    parent_chunk_id: Optional[str]
    text: str
    char_start: int
    char_end: int
    section_heading: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        return int(len(self.text) * settings.tokens_per_char_estimate)


class LegalChunker:
    """Structure-aware chunker for legal documents.

    Strategy:
      1. Split document on legal section headings → sections
      2. Build parent chunks (~2 k chars) for context expansion
      3. Build child chunks (~512 chars with overlap) for retrieval
      4. Each child carries a reference to its parent
    """

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
        min_chunk_size: int = settings.min_chunk_size,
        parent_chunk_size: int = settings.parent_chunk_size,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.parent_chunk_size = parent_chunk_size

    # ── Public ─────────────────────────────────────────────────────────────

    def chunk(self, doc: LoadedDocument) -> tuple[list[Chunk], list[Chunk]]:
        """Return (parent_chunks, child_chunks)."""
        sections = self._split_sections(doc.raw_text)
        parent_chunks: list[Chunk] = []
        child_chunks: list[Chunk] = []

        chunk_idx = 0
        for heading, text, offset in sections:
            parents = self._make_parent_chunks(
                text=text,
                doc_id=doc.doc_id,
                heading=heading,
                base_offset=offset,
                start_idx=chunk_idx,
            )
            for parent in parents:
                parent_chunks.append(parent)
                children = self._make_child_chunks(
                    text=parent.text,
                    doc_id=doc.doc_id,
                    parent_id=parent.chunk_id,
                    heading=heading,
                    base_offset=parent.char_start,
                    start_idx=chunk_idx,
                )
                for child in children:
                    child.metadata.update(doc.metadata)
                    child_chunks.append(child)
                    chunk_idx += 1
            for parent in parents:
                parent.metadata.update(doc.metadata)

        logger.info(
            f"[{doc.file_name}] → {len(parent_chunks)} parents, {len(child_chunks)} children"
        )
        return parent_chunks, child_chunks

    # ── Internals ──────────────────────────────────────────────────────────

    def _split_sections(
        self, text: str
    ) -> list[tuple[str, str, int]]:
        """Return list of (heading, section_text, char_offset)."""
        # Find all heading positions
        heading_spans: list[tuple[int, int, str]] = []
        for pattern in _SECTION_PATTERNS:
            for m in pattern.finditer(text):
                heading_spans.append((m.start(), m.end(), m.group(0).strip()))

        if not heading_spans:
            return [("DOCUMENT", text, 0)]

        # Deduplicate and sort by position
        seen: set[int] = set()
        unique: list[tuple[int, int, str]] = []
        for start, end, heading in sorted(heading_spans):
            if start not in seen:
                seen.add(start)
                unique.append((start, end, heading))

        sections: list[tuple[str, str, int]] = []

        # Preamble before first heading
        first_start = unique[0][0]
        if first_start > 0:
            preamble = text[:first_start].strip()
            if len(preamble) >= self.min_chunk_size:
                sections.append(("PREAMBLE", preamble, 0))

        for i, (start, end, heading) in enumerate(unique):
            next_start = unique[i + 1][0] if i + 1 < len(unique) else len(text)
            section_text = text[end:next_start].strip()
            if len(section_text) >= self.min_chunk_size:
                sections.append((heading, section_text, start))

        return sections if sections else [("DOCUMENT", text, 0)]

    def _make_parent_chunks(
        self,
        text: str,
        doc_id: str,
        heading: str,
        base_offset: int,
        start_idx: int,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        start = 0
        idx = start_idx
        while start < len(text):
            end = min(start + self.parent_chunk_size, len(text))
            piece = text[start:end]
            if len(piece) >= self.min_chunk_size:
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        doc_id=doc_id,
                        parent_chunk_id=None,
                        text=piece,
                        char_start=base_offset + start,
                        char_end=base_offset + end,
                        section_heading=heading,
                        chunk_index=idx,
                    )
                )
                idx += 1
            start += self.parent_chunk_size
        return chunks

    def _make_child_chunks(
        self,
        text: str,
        doc_id: str,
        parent_id: str,
        heading: str,
        base_offset: int,
        start_idx: int,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        start = 0
        idx = start_idx
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            piece = text[start:end].strip()
            if len(piece) >= self.min_chunk_size:
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        doc_id=doc_id,
                        parent_chunk_id=parent_id,
                        text=piece,
                        char_start=base_offset + start,
                        char_end=base_offset + end,
                        section_heading=heading,
                        chunk_index=idx,
                    )
                )
                idx += 1
            if end >= len(text):
                break
            start += self.chunk_size - self.chunk_overlap
        return chunks
