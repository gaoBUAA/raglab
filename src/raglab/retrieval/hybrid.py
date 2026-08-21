"""混合检索：BM25（词法）+ 向量（语义），用 RRF 融合排序。"""

from __future__ import annotations

from raglab.schemas import ScoredChunk
from raglab.storage import KnowledgeBase


class HybridRetriever:
    def __init__(self, kb: KnowledgeBase, rrf_k: int = 60) -> None:
        self.kb = kb
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        """两个检索器各取 top_k*2 候选，再按 RRF 融合为最终 top_k。"""
        candidates = top_k * 2
        bm25_hits = self.kb.search_bm25(query, candidates)
        vector_hits = self.kb.search_vector(query, candidates)
        return self._rrf_fusion([bm25_hits, vector_hits], top_k)

    def _rrf_fusion(self, lists: list[list[ScoredChunk]], top_k: int) -> list[ScoredChunk]:
        """Reciprocal Rank Fusion：
        score(d) = Σ_list 1 / (k + rank(d, list))，
        与具体分值解耦，天然可融合异构检索器。"""
        fused: dict[str, ScoredChunk] = {}
        for results in lists:
            for rank, item in enumerate(results, start=1):
                cid = item.chunk.id
                if cid not in fused:
                    fused[cid] = ScoredChunk(
                        chunk=item.chunk,
                        score=0.0,
                        engine="hybrid",
                    )
                fused[cid].score += 1.0 / (self.rrf_k + rank)
        ranked = sorted(fused.values(), key=lambda x: x.score, reverse=True)
        return ranked[:top_k]
