"""测试夹具：全部使用 mock LLM/Embedding，离线可跑。"""

from __future__ import annotations

import pytest

from raglab.config import Settings
from raglab.pipeline.rag import RAGEngine
from raglab.storage import KnowledgeBase


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        llm_provider="mock",
        embedding_provider="mock",
        embedding_dim=64,
        data_dir=tmp_path / "data",
    )


@pytest.fixture()
def kb(settings) -> KnowledgeBase:
    from raglab.engine import build_knowledge_base

    return build_knowledge_base(settings)


@pytest.fixture()
def engine(settings, kb) -> RAGEngine:
    from raglab.engine import build_llm
    from raglab.retrieval import HybridRetriever

    return RAGEngine(
        kb=kb,
        llm=build_llm(settings),
        retriever=HybridRetriever(kb, rrf_k=settings.rrf_k),
        top_k=settings.top_k,
        max_tokens=settings.llm_max_tokens,
    )
