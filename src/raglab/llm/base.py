"""LLM 抽象层：RAG / Agent / 评测都只依赖 BaseLLM，便于替换与测试。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


class BaseLLM(ABC):
    """所有 LLM 实现的统一接口。"""

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        """messages 遵循 Chat 格式：[{"role": "user", "content": "..."}]"""
