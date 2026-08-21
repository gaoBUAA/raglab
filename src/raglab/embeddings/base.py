"""Embedding 抽象层。"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量编码文本，返回 shape=(n, dim) 的向量列表。"""

    @property
    @abstractmethod
    def dim(self) -> int:
        """向量维度。"""
