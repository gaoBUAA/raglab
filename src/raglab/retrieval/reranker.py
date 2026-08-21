"""重排器：在精排阶段用交叉编码器（或恒等）重新打分。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from raglab.schemas import ScoredChunk


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        """返回重新排序后的列表。"""


class IdentityReranker(Reranker):
    def rerank(self, query: str, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        return chunks


class CrossEncoderReranker(Reranker):
    """BGE-Reranker 交叉编码器（需安装 raglab[local]）。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("请先安装 raglab[local]（sentence-transformers）") from exc
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        pairs = [(query, item.chunk.text) for item in chunks]
        scores = self._model.predict(pairs)
        ranked = [
            ScoredChunk(chunk=item.chunk, score=float(score), engine="rerank")
            for item, score in zip(chunks, scores, strict=True)
        ]
        return sorted(ranked, key=lambda x: x.score, reverse=True)
