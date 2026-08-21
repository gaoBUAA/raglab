"""OpenAI 兼容客户端：支持 DeepSeek / Qwen / Ollama / vLLM 等任意兼容端点。"""

from __future__ import annotations

import time

from raglab.llm.base import BaseLLM, LLMResult


class OpenAICompatLLM(BaseLLM):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        started = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature if temperature is None else temperature,
            max_tokens=self._max_tokens if max_tokens is None else max_tokens,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        return LLMResult(
            text=resp.choices[0].message.content or "",
            prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
            latency_ms=latency_ms,
        )
