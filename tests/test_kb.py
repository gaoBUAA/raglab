from raglab.schemas import ScoredChunk
from raglab.storage import KnowledgeBase


def test_kb_ingest_and_hybrid_search(kb: KnowledgeBase):
    kb.add_document(
        "doc1",
        "公司介绍.md",
        "北航（北京航空航天大学）是新中国第一所航空航天高等学府。",
    )
    kb.add_document("doc2", "其他.md", "今天是晴天，适合去公园散步。")
    hits = kb.search_bm25("北航 航空航天", top_k=3)
    assert hits and hits[0].chunk.document_id == "doc1"
    vec_hits = kb.search_vector("北京航空航天大学", top_k=3)
    assert vec_hits and vec_hits[0].chunk.document_id == "doc1"
    assert kb.count_documents() == 2


def test_rrf_fusion_merges_engines(kb: KnowledgeBase):
    from raglab.retrieval import HybridRetriever

    kb.add_document("doc1", "a.md", "北航成立于1952年，位于北京海淀区学院路。")
    kb.add_document("doc2", "b.md", "上海交通大学位于上海市闵行区。")
    retriever = HybridRetriever(kb)
    results = retriever.retrieve("北航 成立 时间", top_k=2)
    assert isinstance(results[0], ScoredChunk)
    assert results[0].chunk.document_id == "doc1"


def test_kb_persists_across_instances(tmp_path):
    from raglab.embeddings import MockEmbedding

    data_dir = tmp_path / "data"
    kb1 = KnowledgeBase(embedding=MockEmbedding(dim=64), data_dir=data_dir)
    kb1.add_document("doc1", "a.md", "北京航空航天大学位于北京市海淀区学院路37号。")

    kb2 = KnowledgeBase(embedding=MockEmbedding(dim=64), data_dir=data_dir)
    kb2.load()
    assert kb2.count_documents() == 1
    bm25_hits = kb2.search_bm25("北航 学院路", top_k=3)
    assert bm25_hits and bm25_hits[0].chunk.document_id == "doc1"
    vec_hits = kb2.search_vector("北京航空航天大学", top_k=3)
    assert vec_hits and vec_hits[0].chunk.document_id == "doc1"
