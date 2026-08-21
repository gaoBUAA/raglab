"""确定性 Mock LLM：无网络、无密钥，用于单元测试与 CI 冒烟。"""

from __future__ import annotations

from collections.abc import Callable

from raglab.llm.base import BaseLLM, LLMResult

Responder = Callable[[list[dict[str, str]]], str]


class MockLLM(BaseLLM):
    def __init__(self, responder: Responder | None = None) -> None:
        self._responder = responder or self._default_responder

    @staticmethod
    def _default_responder(messages: list[dict[str, str]]) -> str:
        """内置规则：
        - 评测类问题（忠实度/相关性）返回确定性分值；
        - 其余返回用户消息中的第一句，保证 RAG 流程可端到端跑通。
        """
        user_text = "\n".join(m["content"] for m in messages if m["role"] == "user")
        if "忠实" in user_text or "faithfulness" in user_text.lower():
            return "1"
        if "相关" in user_text or "relevance" in user_text.lower():
            return "5"
        first_line = next(
            (line.strip() for line in user_text.splitlines() if line.strip()),
            "Mock 回答",
        )
        return first_line[:120]

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        return LLMResult(text=self._responder(messages))
