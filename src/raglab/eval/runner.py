"""评测运行器：对每个样本跑完整 RAG 流程并聚合指标。"""

from __future__ import annotations

from raglab.embeddings import BaseEmbedding
from raglab.eval.metrics import (
    answer_relevance,
    citation_accuracy,
    context_relevance,
    faithfulness,
    hit_rate_at_k,
    mrr_at_k,
    precision_at_k,
)
from raglab.pipeline.prompts import build_numbered_context
from raglab.pipeline.rag import RAGEngine
from raglab.schemas import EvalReport, EvalSample, EvalScore


class EvalRunner:
    def __init__(self, engine: RAGEngine, embedding: BaseEmbedding | None = None) -> None:
        self.engine = engine
        self.embedding = embedding or engine.kb.embedding

    def run(self, samples: list[EvalSample], top_k: int | None = None) -> EvalReport:
        details: dict[str, list[dict]] = {}
        for sample in samples:
            answer = self.engine.answer(sample.question, top_k=top_k)
            retrieved_ids = [c.chunk.id for c in answer.contexts]
            context = build_numbered_context(answer.contexts, len(answer.contexts))
            golden_ids = self._expand_golden(sample)
            row: dict = {
                "sample_id": sample.id,
                "question": sample.question,
                "answer": answer.answer[:200],
                "retrieved_ids": retrieved_ids,
            }
            row["hit_rate@k"] = hit_rate_at_k(retrieved_ids, golden_ids)
            row["mrr@k"] = mrr_at_k(retrieved_ids, golden_ids)
            row["precision@k"] = precision_at_k(retrieved_ids, golden_ids)
            row["citation_accuracy"] = citation_accuracy(answer.citations, golden_ids)
            row["context_relevance"] = context_relevance(
                sample.question, answer.contexts, self.embedding
            )
            row["faithfulness"] = faithfulness(answer.answer, context, self.engine.llm)
            row["answer_relevance"] = answer_relevance(
                sample.question, answer.answer, self.engine.llm
            )
            row["latency_ms"] = answer.latency_ms
            for key, value in row.items():
                if isinstance(value, float):
                    details.setdefault(key, []).append(round(value, 4))

        scores = [
            EvalScore(
                metric=metric,
                value=round(sum(values) / len(values), 4),
                sample_count=len(values),
            )
            for metric, values in details.items()
        ]
        return EvalReport(
            dataset="custom",
            engine_version="raglab-0.1",
            scores=scores,
            details=details,
        )

    def _expand_golden(self, sample: EvalSample) -> list[str]:
        """金标既可以是 chunk id，也可以是文档 id（自动展开为该文档的全部 chunk）。"""
        expanded: set[str] = set(sample.golden_chunk_ids)
        for doc in self.engine.kb.document_store.list_documents():
            if doc.id in expanded:
                expanded.discard(doc.id)
                expanded.update(c.id for c in self.engine.kb.document_store.get_chunks(doc.id))
        return list(expanded)
