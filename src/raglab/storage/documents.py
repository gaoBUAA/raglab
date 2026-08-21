"""SQLite 文档/分块元数据存储。"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from raglab.schemas import Chunk, Document


class DocumentStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id),
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
                """
            )
            self._conn.commit()

    def add_document(self, doc: Document) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO documents (id, source, metadata) VALUES (?, ?, ?)",
                (doc.id, doc.source, json.dumps(doc.metadata, ensure_ascii=False)),
            )
            self._conn.commit()

    def delete_document(self, doc_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
            self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            self._conn.commit()

    def list_documents(self) -> list[Document]:
        with self._lock:
            rows = self._conn.execute("SELECT id, source, metadata FROM documents").fetchall()
            return [
                Document(id=r["id"], source=r["source"], metadata=json.loads(r["metadata"]))
                for r in rows
            ]

    def add_chunks(self, chunks: list[Chunk]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO chunks "
                "(id, document_id, chunk_index, text, metadata) VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        c.id,
                        c.document_id,
                        c.index,
                        c.text,
                        json.dumps(c.metadata, ensure_ascii=False),
                    )
                    for c in chunks
                ],
            )
            self._conn.commit()

    def delete_chunks(self, doc_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
            self._conn.commit()

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
            return self._row_to_chunk(row) if row else None

    def get_chunks(self, doc_id: str | None = None) -> list[Chunk]:
        with self._lock:
            if doc_id is None:
                rows = self._conn.execute(
                    "SELECT * FROM chunks ORDER BY document_id, chunk_index"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index", (doc_id,)
                ).fetchall()
            return [self._row_to_chunk(r) for r in rows]

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            id=row["id"],
            document_id=row["document_id"],
            index=row["chunk_index"],
            text=row["text"],
            metadata=json.loads(row["metadata"]),
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
