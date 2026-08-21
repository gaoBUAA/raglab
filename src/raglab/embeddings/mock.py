"""字符 n-gram 哈希向量：确定性、零依赖，用于离线测试。"""

from __future__ import annotations

import hashlib

import numpy as np

from raglab.embeddings.base import BaseEmbedding


class MockEmbedding(BaseEmbedding):
    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = np.zeros(self._dim, dtype=np.float32)
            for n in (1, 2, 3):
                grams = (text[i : i + n] for i in range(len(text) - n + 1))
                for gram in grams:
                    digest = hashlib.md5(gram.encode("utf-8")).hexdigest()
                    idx = int(digest[:8], 16) % self._dim
                    vec[idx] += 1.0
            norm = float(np.linalg.norm(vec))
            vectors.append((vec / norm).tolist() if norm > 0 else vec.tolist())
        return vectors

    @property
    def dim(self) -> int:
        return self._dim
