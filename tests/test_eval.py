from raglab.eval.dataset import load_dataset
from raglab.eval.metrics import citation_accuracy, hit_rate_at_k, mrr_at_k
from raglab.eval.runner import EvalRunner
from raglab.pipeline.rag import RAGEngine
from raglab.schemas import EvalSample
from raglab.storage import KnowledgeBase


def test_retrieval_metrics():
    assert hit_rate_at_k(["a", "b"], ["b"], k=2) == 1.0
    assert mrr_at_k(["a", "b", "c"], ["c"]) == 1 / 3
    assert mrr_at_k(["a", "b"], ["z"]) == 0.0
    assert citation_accuracy(["a", "b"], ["a", "c"]) == 0.5


def test_eval_runner_produces_report(engine: RAGEngine, kb: KnowledgeBase):
    kb.add_document(
        "buaa",
        "北航.md",
        "北京航空航天大学成立于1952年，校训是德才兼备知行合一，位于北京市海淀区。",
    )
    golden = [c.id for c in kb.all_chunks() if c.document_id == "buaa"]
    samples = [
        EvalSample(id="q1", question="北航的校训是什么？", golden_chunk_ids=golden),
        EvalSample(id="q2", question="北航成立于哪一年？", golden_chunk_ids=golden),
    ]
    report = EvalRunner(engine).run(samples)
    metrics = report.summary()
    assert metrics["hit_rate@k"] == 1.0
    assert metrics["mrr@k"] == 1.0
    assert "faithfulness" in metrics
    assert "answer_relevance" in metrics


def test_load_dataset(tmp_path):
    import json

    path = tmp_path / "dataset.json"
    path.write_text(
        json.dumps({"samples": [{"id": "q1", "question": "你好"}]}),
        encoding="utf-8",
    )
    samples = load_dataset(path)
    assert samples[0].question == "你好"
