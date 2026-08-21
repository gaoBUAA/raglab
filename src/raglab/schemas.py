"""核心数据模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """一份被摄入的知识文档。"""

    id: str
    source: str  # 文件名 / URL / 标题
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """文档切分后的最小检索单元。"""

    id: str
    document_id: str
    text: str
    index: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def source(self) -> str:
        return str(self.metadata.get("source") or self.document_id)


class ScoredChunk(BaseModel):
    """带得分的检索结果。"""

    chunk: Chunk
    score: float
    engine: str = "hybrid"  # bm25 | vector | hybrid | rerank


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    use_agent: bool = False


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[str] = Field(default_factory=list)  # 被引用的 chunk id
    contexts: list[ScoredChunk] = Field(default_factory=list)
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    trace: list[dict[str, Any]] = Field(default_factory=list)


class EvalSample(BaseModel):
    """一条评测样本。golden_chunk_ids 用于检索指标，ground_truth 用于答案指标。"""

    id: str
    question: str
    ground_truth: str | None = None
    golden_chunk_ids: list[str] = Field(default_factory=list)


class EvalScore(BaseModel):
    metric: str
    value: float
    sample_count: int


class EvalReport(BaseModel):
    dataset: str
    engine_version: str
    scores: list[EvalScore] = Field(default_factory=list)
    details: dict[str, list[float]] = Field(default_factory=dict)

    def summary(self) -> dict[str, float]:
        return {s.metric: round(s.value, 4) for s in self.scores}
