"""RAG 主流水线：检索 → 重排 → 生成（带引用）→ 后处理。"""

from __future__ import annotations

import re
import time

from raglab.llm import BaseLLM
from raglab.pipeline.prompts import build_numbered_context, build_rag_messages
from raglab.retrieval import HybridRetriever, IdentityReranker, Reranker
from raglab.schemas import Answer, ScoredChunk
from raglab.storage import KnowledgeBase

_CITE_RE = re.compile(r"\[(\d+)\]")


class RAGEngine:
    def __init__(
        self,
        kb: KnowledgeBase,
        llm: BaseLLM,
        retriever: HybridRetriever | None = None,
        reranker: Reranker | None = None,
        top_k: int = 5,
        max_tokens: int | None = None,
    ) -> None:
        self.kb = kb
        self.llm = llm
        self.retriever = retriever or HybridRetriever(kb)
        self.reranker = reranker or IdentityReranker()
        self.top_k = top_k
        self.max_tokens = max_tokens

    def answer(self, question: str, top_k: int | None = None) -> Answer:
        k = top_k or self.top_k
        started = time.perf_counter()

        retrieved = self.retriever.retrieve(question, k)
        retrieved = self.reranker.rerank(question, retrieved)
        context = build_numbered_context(retrieved, k)

        messages = build_rag_messages(context, question)
        result = self.llm.complete(messages, max_tokens=self.max_tokens)
        citations = self._extract_citations(result.text, retrieved)

        return Answer(
            question=question,
            answer=result.text.strip(),
            citations=citations,
            contexts=retrieved,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            trace=[
                {
                    "step": "retrieve",
                    "top_k": len(retrieved),
                    "engines": sorted({c.engine for c in retrieved}),
                },
                {"step": "generate", "model_tokens": result.completion_tokens},
            ],
        )

    @staticmethod
    def _extract_citations(text: str, contexts: list[ScoredChunk]) -> list[str]:
        """把回答中的 [n] 映射到实际 chunk id（1-based），越界的忽略。"""
        chunk_ids: list[str] = []
        seen: set[str] = set()
        for m in _CITE_RE.finditer(text):
            n = int(m.group(1))
            if 1 <= n <= len(contexts):
                cid = contexts[n - 1].chunk.id
                if cid not in seen:
                    seen.add(cid)
                    chunk_ids.append(cid)
        return chunk_ids
