"""KnowledgeBase：把分块、向量化、双路存储串起来的外观类。"""

from __future__ import annotations

from pathlib import Path

from raglab.chunking import Chunker, FixedSizeChunker
from raglab.embeddings import BaseEmbedding
from raglab.schemas import Chunk, Document, ScoredChunk
from raglab.storage.bm25 import BM25Index
from raglab.storage.documents import DocumentStore
from raglab.storage.vectors import VectorStore


class KnowledgeBase:
    def __init__(
        self,
        embedding: BaseEmbedding,
        data_dir: Path = Path("data"),
        chunker: Chunker | None = None,
        bm25: BM25Index | None = None,
        vector_store: VectorStore | None = None,
        document_store: DocumentStore | None = None,
    ) -> None:
        data_dir = Path(data_dir)
        self.embedding = embedding
        self.chunker = chunker or FixedSizeChunker()
        self.bm25 = bm25 or BM25Index()
        self.vector_store = vector_store or VectorStore(embedding.dim)
        self.document_store = document_store or DocumentStore(data_dir / "raglab.db")
        self._index_dir = data_dir / "vector_index"

    def add_document(
        self, doc_id: str, source: str, text: str, metadata: dict | None = None
    ) -> int:
        """摄入一份文档，返回 chunk 数量。"""
        doc = Document(id=doc_id, source=source, metadata=metadata or {})
        chunks = self.chunker.chunk(doc, text)
        vectors = self.embedding.embed([c.text for c in chunks]) if chunks else []
        self.document_store.add_document(doc)
        self.document_store.delete_chunks(doc_id)  # 幂等：重复摄入先清旧块
        self.document_store.add_chunks(chunks)
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.bm25.add(chunk.id, chunk.text)
            self.vector_store.add(chunk, vector)
        self.save()  # 自动持久化，跨进程可恢复
        return len(chunks)

    def search_bm25(self, query: str, top_k: int) -> list[ScoredChunk]:
        hits = self.bm25.search(query, top_k)
        return [
            ScoredChunk(chunk=chunk, score=score, engine="bm25")
            for chunk_id, score in hits
            if (chunk := self.document_store.get_chunk(chunk_id)) is not None
        ]

    def search_vector(self, query: str, top_k: int) -> list[ScoredChunk]:
        vector = self.embedding.embed([query])[0]
        return self.vector_store.search(vector, top_k)

    def all_chunks(self) -> list[Chunk]:
        return self.document_store.get_chunks()

    def count_documents(self) -> int:
        return len(self.document_store.list_documents())

    def save(self) -> None:
        self.vector_store.save(self._index_dir)

    def load(self) -> None:
        """从磁盘恢复向量索引，并从元数据库重建 BM25 索引。"""
        if (self._index_dir / "matrix.npy").exists():
            self.vector_store.load(self._index_dir)
        for chunk in self.document_store.get_chunks():
            self.bm25.add(chunk.id, chunk.text)
