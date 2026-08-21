"""RAGLab 命令行入口。"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from raglab.agent import Calculator, KnowledgeBaseSearch, ReActAgent
from raglab.config import Settings
from raglab.engine import build_engine
from raglab.eval.dataset import load_dataset
from raglab.eval.report import render_markdown
from raglab.eval.runner import EvalRunner

app = typer.Typer(help="RAGLab - 评测驱动的 RAG + Agent 应用实验室", no_args_is_help=True)


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="知识文档路径（.md/.txt）"),
    doc_id: str = typer.Option(None, help="文档 ID，默认用文件名"),
) -> None:
    """摄入一份文档到知识库。"""
    settings = Settings()
    engine = build_engine(settings)
    text = Path(path).read_text(encoding="utf-8")
    did = doc_id or Path(path).stem
    chunks = engine.kb.add_document(did, Path(path).name, text)
    typer.echo(f"已摄入 {did}，共 {chunks} 个 chunk")


@app.command()
def query(
    question: str = typer.Argument(..., help="问题"),
    top_k: int = typer.Option(None, help="检索数量"),
    agent: bool = typer.Option(False, "--agent", help="使用 ReAct Agent"),
) -> None:
    """向知识库提问。"""
    settings = Settings()
    engine = build_engine(settings)
    if agent:
        react = ReActAgent(
            llm=engine.llm,
            tools=[Calculator(), KnowledgeBaseSearch(engine.retriever, top_k=3)],
            max_iterations=settings.agent_max_iterations,
        )
        typer.echo(react.run(question))
        return
    answer = engine.answer(question, top_k=top_k)
    typer.echo(answer.answer)
    if answer.citations:
        typer.echo("\n引用：" + ", ".join(answer.citations))


@app.command()
def eval_dataset(
    dataset: Path = typer.Argument(..., help="评测数据集 JSON 路径"),
    output: Path = typer.Option(Path("reports/eval_report.md"), "--output", help="报告输出路径"),
    json_output: Path = typer.Option(None, "--json", help="JSON 结果输出路径"),
) -> None:
    """运行评测并生成报告。"""
    engine = build_engine()
    samples = load_dataset(dataset)
    report = EvalRunner(engine).run(samples)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(report), encoding="utf-8")
    typer.echo(f"评测完成：{len(samples)} 条样本")
    for score in report.scores:
        typer.echo(f"  {score.metric}: {score.value:.4f}")
    typer.echo(f"报告已写入 {output}")
    if json_output:
        json_output.write_text(
            json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="监听地址"),
    port: int = typer.Option(8000, help="监听端口"),
) -> None:
    """启动 FastAPI 服务。"""
    import uvicorn

    from raglab.api.server import create_app

    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    app()
