"""分块策略抽象。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from raglab.schemas import Chunk, Document


class Chunker(ABC):
    @abstractmethod
    def chunk(self, document: Document, text: str) -> list[Chunk]:
        """把文档正文切分为 Chunk 列表。"""
