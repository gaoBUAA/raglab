from raglab.storage.bm25 import BM25Index, tokenize


def test_tokenize_mixed_cn_en():
    tokens = tokenize("RAG 是什么？检索增强生成")
    assert "rag" in tokens
    assert "检索" in tokens


def test_bm25_ranks_relevant_first():
    index = BM25Index()
    index.add("a", "北京大学的计算机学院位于北京市海淀区")
    index.add("b", "上海的天气今天很好，适合出游")
    hits = index.search("北京大学计算机学院", top_k=2)
    assert hits[0][0] == "a"


def test_bm25_empty_query():
    index = BM25Index()
    index.add("a", "随便一段话")
    assert index.search("", top_k=5) == []
