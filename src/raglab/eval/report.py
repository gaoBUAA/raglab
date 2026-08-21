"""把评测结果渲染为 Markdown 报告。"""

from __future__ import annotations

from raglab.schemas import EvalReport


def render_markdown(report: EvalReport) -> str:
    lines = [
        "# RAGLab 评测报告",
        "",
        f"- 数据集：`{report.dataset}`",
        f"- 引擎版本：`{report.engine_version}`",
        "",
        "## 指标汇总",
        "",
        "| 指标 | 均值 | 样本数 |",
        "| --- | --- | --- |",
    ]
    for score in report.scores:
        lines.append(f"| {score.metric} | {score.value:.4f} | {score.sample_count} |")
    lines += ["", "## 逐样本明细", ""]
    for metric, values in report.details.items():
        lines.append(f"### {metric}")
        lines.append("")
        lines.append("| # | 值 |")
        lines.append("| --- | --- |")
        for i, value in enumerate(values, start=1):
            lines.append(f"| {i} | {value:.4f} |")
        lines.append("")
    return "\n".join(lines)
