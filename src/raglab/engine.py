"""组装工厂：根据 Settings 构建完整的 RAGEngine。"""

from __future__ import annotations

from raglab.config import Settings
from raglab.embeddings import MockEmbedding, OpenAICompatEmbedding
from raglab.llm import MockLLM, OpenAICompatLLM
from raglab.pipeline.rag import RAGEngine
from raglab.retrieval import CrossEncoderReranker, HybridRetriever
from raglab.storage import KnowledgeBase


def build_llm(settings: Settings):
    if settings.llm_provider == "mock":
        return MockLLM()
    return OpenAICompatLLM(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )


def build_embedding(settings: Settings):
    if settings.embedding_provider == "mock":
        return MockEmbedding(dim=settings.embedding_dim)
    return OpenAICompatEmbedding(
        base_url=settings.llm_base_url,
        api_key=settings.effective_embedding_api_key,
        model=settings.embedding_model,
        dim=settings.embedding_dim,
    )


def build_knowledge_base(settings: Settings) -> KnowledgeBase:
    kb = KnowledgeBase(embedding=build_embedding(settings), data_dir=settings.data_dir)
    kb.load()
    return kb


def build_engine(settings: Settings | None = None) -> RAGEngine:
    settings = settings or Settings()
    kb = build_knowledge_base(settings)
    reranker = CrossEncoderReranker(settings.reranker_model) if settings.use_reranker else None
    return RAGEngine(
        kb=kb,
        llm=build_llm(settings),
        retriever=HybridRetriever(kb, rrf_k=settings.rrf_k),
        reranker=reranker,
        top_k=settings.top_k,
        max_tokens=settings.llm_max_tokens,
    )
