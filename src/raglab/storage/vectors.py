"""基于 NumPy 的内存向量检索：余弦相似度，支持持久化到磁盘。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from raglab.schemas import Chunk, ScoredChunk


class VectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._matrix: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._rows: list[tuple[str, Chunk]] = []

    @property
    def size(self) -> int:
        return len(self._rows)

    def add(self, chunk: Chunk, vector: list[float]) -> None:
        self._matrix = np.vstack(
            [self._matrix, np.asarray(vector, dtype=np.float32).reshape(1, -1)]
        )
        self._rows.append((chunk.id, chunk))

    def search(self, query_vector: list[float], top_k: int = 5) -> list[ScoredChunk]:
        if self.size == 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        scores = (self._matrix @ q) / (np.linalg.norm(self._matrix, axis=1) * q_norm + 1e-9)
        order = np.argsort(scores)[::-1][:top_k]
        return [
            ScoredChunk(chunk=self._rows[i][1], score=float(scores[i]), engine="vector")
            for i in order
        ]

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "matrix.npy", self._matrix)
        with (index_dir / "rows.json").open("w", encoding="utf-8") as f:
            json.dump(
                [{"id": cid, "chunk": chunk.model_dump(mode="json")} for cid, chunk in self._rows],
                f,
                ensure_ascii=False,
            )

    def load(self, index_dir: Path) -> None:
        self._matrix = np.load(index_dir / "matrix.npy")
        with (index_dir / "rows.json").open("r", encoding="utf-8") as f:
            rows = json.load(f)
        self._rows = [(r["id"], Chunk.model_validate(r["chunk"])) for r in rows]
