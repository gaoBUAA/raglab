from raglab.pipeline.rag import RAGEngine
from raglab.schemas import Chunk, ScoredChunk
from raglab.storage import KnowledgeBase


def test_rag_answer_with_citations(engine: RAGEngine, kb: KnowledgeBase):
    kb.add_document(
        "buaa",
        "北航.md",
        "北京航空航天大学成立于1952年10月25日，是中国第一所航空航天高等学府，"
        "校训为“德才兼备、知行合一”。",
    )
    answer = engine.answer("北航成立于哪一年？")
    assert answer.question == "北航成立于哪一年？"
    assert "1952" in answer.answer or answer.answer
    assert answer.contexts
    assert answer.latency_ms > 0


def test_citation_extraction_maps_to_chunks():
    chunk = Chunk(id="d#0", document_id="d", text="北京航空航天大学成立于1952年", index=0)
    ctx = [ScoredChunk(chunk=chunk, score=1.0)]
    cited = RAGEngine._extract_citations("北航成立于[1]1952年[2]。", ctx)
    assert cited == ["d#0"]
