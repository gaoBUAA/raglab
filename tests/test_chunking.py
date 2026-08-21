from raglab.chunking import FixedSizeChunker, MarkdownChunker
from raglab.schemas import Document


def test_fixed_size_chunker_respects_overlap():
    chunker = FixedSizeChunker(chunk_size=20, overlap=5)
    doc = Document(id="d1", source="a.md")
    chunks = chunker.chunk(doc, "0123456789" * 4)
    assert len(chunks) > 1
    assert all(c.id.startswith("d1#") for c in chunks)
    # 相邻 chunk 应共享 overlap 部分
    assert chunks[0].text[-5:] in chunks[1].text[:5] or chunks[1].text[:5] in chunks[0].text[-5:]


def test_markdown_chunker_keeps_headings():
    chunker = MarkdownChunker()
    doc = Document(id="d2", source="b.md")
    text = "# 第一章\n内容一\n\n## 第一节\n内容二\n\n# 第二章\n内容三"
    chunks = chunker.chunk(doc, text)
    assert len(chunks) >= 3
    assert any("# 第一章" in c.text for c in chunks)
    assert any("# 第二章" in c.text for c in chunks)


def test_chunker_empty_text():
    chunker = FixedSizeChunker()
    doc = Document(id="d3", source="c.md")
    assert chunker.chunk(doc, "") == []
