from fastapi.testclient import TestClient

from raglab.api.server import create_app
from raglab.pipeline.rag import RAGEngine
from raglab.storage import KnowledgeBase


def test_api_health_ingest_query(engine: RAGEngine, kb: KnowledgeBase):
    client = TestClient(create_app(engine=engine))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    ingest = client.post(
        "/ingest",
        json={"document_id": "doc1", "source": "a.md", "text": "北京航空航天大学成立于1952年。"},
    )
    assert ingest.status_code == 200
    assert ingest.json()["chunks"] >= 1

    query = client.post("/query", json={"question": "北航成立时间"})
    assert query.status_code == 200
    assert query.json()["question"] == "北航成立时间"

    agent = client.post("/agent", json={"question": "计算 3*7 等于多少"})
    assert agent.status_code == 200
    assert "answer" in agent.json()
