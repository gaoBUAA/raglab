"""评测指标：离线可计算的检索指标 + LLM-as-judge 答案指标。"""

from __future__ import annotations

import re

from raglab.embeddings import BaseEmbedding
from raglab.llm import BaseLLM
from raglab.schemas import ScoredChunk

_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+")


def hit_rate_at_k(retrieved_ids: list[str], golden_ids: list[str], k: int | None = None) -> float:
    """检索结果中是否命中任意金标 chunk（0/1），在数据集上取均值即 recall@k 的常用替代。"""
    k = k or len(retrieved_ids)
    return 1.0 if any(cid in golden_ids for cid in retrieved_ids[:k]) else 0.0


def mrr_at_k(retrieved_ids: list[str], golden_ids: list[str]) -> float:
    """第一个命中金标 chunk 的位置倒数；未命中为 0。"""
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in golden_ids:
            return 1.0 / rank
    return 0.0


def precision_at_k(retrieved_ids: list[str], golden_ids: list[str], k: int | None = None) -> float:
    """检索结果中金标 chunk 的占比。"""
    k = k or len(retrieved_ids)
    hits = sum(1 for cid in retrieved_ids[:k] if cid in golden_ids)
    return hits / k if k else 0.0


def citation_accuracy(cited_ids: list[str], golden_ids: list[str]) -> float:
    """被引用的 chunk 中，有多少比例指向金标资料。"""
    if not cited_ids:
        return 0.0
    return sum(1 for cid in cited_ids if cid in golden_ids) / len(cited_ids)


def context_relevance(query: str, contexts: list[ScoredChunk], embedding: BaseEmbedding) -> float:
    """查询向量与检索上下文向量的平均余弦相似度（语义相关性代理指标）。"""
    if not contexts:
        return 0.0
    q_vec = embedding.embed([query])[0]
    c_vecs = embedding.embed([c.chunk.text for c in contexts])
    return _cosine(q_vec, c_vecs)


def faithfulness(answer: str, context: str, llm: BaseLLM) -> float:
    """忠实度：回答是否被上下文支持（LLM-as-judge，0~1）。"""
    prompt = (
        f"【上下文】\n{context}\n\n【回答】\n{answer}\n\n"
        "判断回答中的陈述是否都能从上下文找到依据。只输出一个 0 到 1 之间的数字，"
        "1 表示完全忠实，0 表示完全编造。"
    )
    out = llm.complete([{"role": "user", "content": prompt}], temperature=0.0).text
    return _clamp(_parse_number(out), 0.0, 1.0)


def answer_relevance(question: str, answer: str, llm: BaseLLM) -> float:
    """答案相关性：回答是否切题（LLM-as-judge，0~5）。"""
    prompt = (
        f"【问题】\n{question}\n\n【回答】\n{answer}\n\n"
        "从 0 到 5 给回答与问题的相关性打分，只输出一个数字。"
    )
    out = llm.complete([{"role": "user", "content": prompt}], temperature=0.0).text
    return _clamp(_parse_number(out), 0.0, 5.0)


def _parse_number(text: str) -> float:
    match = _NUMBER_RE.search(text)
    return float(match.group()) if match else 0.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _cosine(q_vec: list[float], c_vecs: list[list[float]]) -> float:
    import math

    q_norm = math.sqrt(sum(x * x for x in q_vec))
    if q_norm == 0:
        return 0.0
    total = 0.0
    for vec in c_vecs:
        dot = sum(a * b for a, b in zip(q_vec, vec, strict=True))
        c_norm = math.sqrt(sum(x * x for x in vec))
        total += dot / (q_norm * c_norm) if c_norm else 0.0
    return total / len(c_vecs)
