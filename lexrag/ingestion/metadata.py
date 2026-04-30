from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

from config import settings
from lexrag.utils.logging import get_logger

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    file_path   TEXT NOT NULL,
    file_name   TEXT NOT NULL,
    file_type   TEXT NOT NULL,
    char_count  INTEGER,
    chunk_count INTEGER,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata    TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id         TEXT PRIMARY KEY,
    doc_id           TEXT NOT NULL REFERENCES documents(doc_id),
    parent_chunk_id  TEXT,
    chunk_index      INTEGER NOT NULL,
    char_start       INTEGER NOT NULL,
    char_end         INTEGER NOT NULL,
    section_heading  TEXT,
    text             TEXT NOT NULL,
    metadata         TEXT DEFAULT '{}',
    created_at       DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_parent  ON chunks(parent_chunk_id);
"""


class MetadataStore:
    """SQLite-backed store for document and chunk metadata."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or settings.db_path
        self._init_db()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
        logger.debug(f"SQLite DB ready at {self._db_path}")

    # ── Documents ──────────────────────────────────────────────────────────

    def upsert_document(
        self,
        doc_id: str,
        file_path: str,
        file_name: str,
        file_type: str,
        char_count: int,
        chunk_count: int,
        metadata: dict,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO documents
                    (doc_id, file_path, file_name, file_type, char_count, chunk_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    file_path   = excluded.file_path,
                    file_name   = excluded.file_name,
                    char_count  = excluded.char_count,
                    chunk_count = excluded.chunk_count,
                    metadata    = excluded.metadata
                """,
                (
                    doc_id,
                    file_path,
                    file_name,
                    file_type,
                    char_count,
                    chunk_count,
                    json.dumps(metadata),
                ),
            )

    def get_document(self, doc_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if row:
                d = dict(row)
                d["metadata"] = json.loads(d["metadata"])
                return d
        return None

    def list_documents(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
            result = []
            for row in rows:
                d = dict(row)
                d["metadata"] = json.loads(d["metadata"])
                result.append(d)
            return result

    def document_exists(self, doc_id: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            return row is not None

    # ── Chunks ─────────────────────────────────────────────────────────────

    def insert_chunks(self, chunks: list[Any]) -> None:
        """Insert Chunk objects."""
        with self._conn() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunks
                    (chunk_id, doc_id, parent_chunk_id, chunk_index,
                     char_start, char_end, section_heading, text, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c.chunk_id,
                        c.doc_id,
                        c.parent_chunk_id,
                        c.chunk_index,
                        c.char_start,
                        c.char_end,
                        c.section_heading,
                        c.text,
                        json.dumps(c.metadata),
                    )
                    for c in chunks
                ],
            )

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
            if row:
                d = dict(row)
                d["metadata"] = json.loads(d["metadata"])
                return d
        return None

    def get_parent_chunk(self, child_chunk_id: str) -> Optional[dict]:
        with self._conn() as conn:
            child = conn.execute(
                "SELECT parent_chunk_id FROM chunks WHERE chunk_id = ?",
                (child_chunk_id,),
            ).fetchone()
            if child and child["parent_chunk_id"]:
                return self.get_chunk(child["parent_chunk_id"])
        return None

    def get_chunks_by_doc(self, doc_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index",
                (doc_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_all_child_chunks(self) -> list[dict]:
        """Return chunks that have a parent (i.e., are child/retrieval chunks)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE parent_chunk_id IS NOT NULL ORDER BY doc_id, chunk_index"
            ).fetchall()
            result = []
            for row in rows:
                d = dict(row)
                d["metadata"] = json.loads(d["metadata"])
                result.append(d)
            return result

    def get_total_chunk_count(self) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM chunks WHERE parent_chunk_id IS NOT NULL"
            ).fetchone()
            return row["cnt"] if row else 0

    def delete_document(self, doc_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
