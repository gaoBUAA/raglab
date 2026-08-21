"""OpenAI 兼容 Embedding：DeepSeek / Qwen / Ollama / 本地 OpenAI 兼容服务。"""

from __future__ import annotations

from raglab.embeddings.base import BaseEmbedding


class OpenAICompatEmbedding(BaseEmbedding):
    def __init__(self, base_url: str, api_key: str, model: str, dim: int = 1536) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
        self._model = model
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in resp.data]

    @property
    def dim(self) -> int:
        return self._dim
