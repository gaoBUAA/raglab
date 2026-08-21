"""FastAPI 服务：摄入、问答、Agent、评测。"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from raglab import __version__
from raglab.agent import Calculator, KnowledgeBaseSearch, ReActAgent
from raglab.config import Settings
from raglab.engine import build_engine
from raglab.eval.runner import EvalRunner
from raglab.pipeline.rag import RAGEngine
from raglab.schemas import EvalSample, QueryRequest


class IngestRequest(BaseModel):
    document_id: str
    source: str
    text: str


class EvalRequest(BaseModel):
    samples: list[EvalSample]


def create_app(engine: RAGEngine | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    engine = engine or build_engine(settings)
    app = FastAPI(title="RAGLab", version=__version__, description="Eval-first RAG & Agent lab")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__, "documents": engine.kb.count_documents()}

    @app.post("/ingest")
    def ingest(req: IngestRequest) -> dict:
        chunks = engine.kb.add_document(req.document_id, req.source, req.text)
        return {"document_id": req.document_id, "chunks": chunks}

    @app.post("/ingest/file")
    async def ingest_file(file: UploadFile) -> dict:
        raw = await file.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail="仅支持 UTF-8 文本文件") from exc
        doc_id = Path(file.filename or "upload").stem
        chunks = engine.kb.add_document(doc_id, file.filename or doc_id, text)
        return {"document_id": doc_id, "chunks": chunks}

    @app.post("/query")
    def query(req: QueryRequest) -> dict:
        answer = engine.answer(req.question, top_k=req.top_k)
        return answer.model_dump(mode="json")

    @app.post("/agent")
    def agent_query(req: QueryRequest) -> dict:
        agent = ReActAgent(
            llm=engine.llm,
            tools=[Calculator(), KnowledgeBaseSearch(engine.retriever, top_k=3)],
            max_iterations=settings.agent_max_iterations,
        )
        final = agent.run(req.question)
        return {"question": req.question, "answer": final, "trace": agent.trace}

    @app.post("/eval")
    def run_eval(req: EvalRequest) -> dict:
        report = EvalRunner(engine).run(req.samples, top_k=settings.top_k)
        return report.model_dump(mode="json")

    return app
