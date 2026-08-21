"""常用分块策略：固定长度、递归字符、Markdown 标题感知。"""

from __future__ import annotations

import re

from raglab.chunking.base import Chunker
from raglab.schemas import Chunk, Document


def _chunk_id(doc_id: str, index: int) -> str:
    return f"{doc_id}#{index}"


class FixedSizeChunker(Chunker):
    """按固定字符数切分，带 overlap 避免割断语义。"""

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap 必须小于 chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document, text: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        for i, start in enumerate(range(0, max(len(text), 1), step)):
            piece = text[start : start + self.chunk_size].strip()
            if piece:
                chunks.append(
                    Chunk(
                        id=_chunk_id(document.id, i),
                        document_id=document.id,
                        text=piece,
                        index=i,
                        metadata={"source": document.source},
                    )
                )
        return chunks


class RecursiveCharacterChunker(Chunker):
    """按分隔符优先级递归切分（参考 LangChain 思路，但逻辑自实现）。"""

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        separators: tuple[str, ...] = ("\n\n", "\n", "。", "；", " ", ""),
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators

    def chunk(self, document: Document, text: str) -> list[Chunk]:
        pieces = self._split(text, self.separators)
        chunks: list[Chunk] = []
        buffer = ""
        for piece in pieces:
            if len(buffer) + len(piece) <= self.chunk_size:
                buffer += piece
            else:
                if buffer.strip():
                    chunks.append(buffer.strip())
                buffer = piece
        if buffer.strip():
            chunks.append(buffer.strip())
        return [
            Chunk(
                id=_chunk_id(document.id, i),
                document_id=document.id,
                text=text,
                index=i,
                metadata={"source": document.source},
            )
            for i, text in enumerate(chunks)
        ]

    def _split(self, text: str, separators: tuple[str, ...]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        sep = separators[0]
        parts = text.split(sep) if sep else list(text)
        if len(parts) == 1:
            return self._split(text, separators[1:]) if len(separators) > 1 else [text]
        result: list[str] = []
        for part in parts:
            result.extend(self._split(part, separators[1:]) if len(separators) > 1 else [part])
        return result


class MarkdownChunker(Chunker):
    """按 Markdown 标题切分，保留标题上下文；过长的段落交给递归切分。"""

    _HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._fallback = RecursiveCharacterChunker(chunk_size=chunk_size, overlap=overlap)

    def chunk(self, document: Document, text: str) -> list[Chunk]:
        matches = list(self._HEADING_RE.finditer(text))
        if not matches:
            return self._fallback.chunk(document, text)
        sections: list[str] = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections.append(text[start:end].strip())
        chunks: list[Chunk] = []
        for section in sections:
            if len(section) <= self.chunk_size:
                chunks.append(section)
            else:
                chunks.extend(c.text for c in self._fallback.chunk(document, section))
        return [
            Chunk(
                id=_chunk_id(document.id, i),
                document_id=document.id,
                text=text,
                index=i,
                metadata={"source": document.source},
            )
            for i, text in enumerate(chunks)
        ]
