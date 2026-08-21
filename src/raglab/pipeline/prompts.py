"""提示词模板集中管理，方便评测与迭代。"""

from __future__ import annotations

RAG_SYSTEM_PROMPT = (
    "你是一个严谨的问答助手。请只依据下面提供的【上下文】回答问题，"
    "不要使用外部知识。引用格式：在相关句末标注来源编号，例如 [1][2]。"
    "如果上下文中没有答案，请明确回答“根据提供的资料无法回答”。"
)


def build_rag_messages(context: str, question: str) -> list[dict[str, str]]:
    user = f"【上下文】\n{context}\n\n【问题】\n{question}\n\n请回答，并标注引用编号。"
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_numbered_context(chunks: list, k: int = 5) -> str:
    """把检索结果渲染成带编号的上下文。chunks 为 ScoredChunk 列表。"""
    lines = []
    for i, item in enumerate(chunks[:k], start=1):
        lines.append(f"[{i}]（来源：{item.chunk.source}）\n{item.chunk.text}")
    return "\n\n".join(lines)
