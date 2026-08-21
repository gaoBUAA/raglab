"""Agent 工具集：每个工具都有名称、描述和 JSON Schema，便于 LLM 调用。"""

from __future__ import annotations

import ast
import operator
from abc import ABC, abstractmethod

from raglab.pipeline.prompts import build_numbered_context
from raglab.retrieval import HybridRetriever


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    parameters: dict = {}

    @abstractmethod
    def run(self, **kwargs) -> str:
        """执行工具，返回可读文本结果。"""

    def to_spec(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


class Calculator(BaseTool):
    name = "calculator"
    description = "对四则运算/幂运算的数学表达式求值，例如 (2+3)*4。"
    parameters = {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "数学表达式"}},
        "required": ["expression"],
    }

    _ALLOWED = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
    }

    def run(self, expression: str) -> str:
        value = self._safe_eval(expression)
        return f"{expression} = {value}"

    def _safe_eval(self, expression: str) -> float:
        tree = ast.parse(expression, mode="eval")

        def eval_node(node: ast.AST):
            if isinstance(node, ast.Expression):
                return eval_node(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in self._ALLOWED:
                return self._ALLOWED[type(node.op)](eval_node(node.left), eval_node(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in self._ALLOWED:
                return self._ALLOWED[type(node.op)](eval_node(node.operand))
            raise ValueError(f"不支持的表达式: {expression}")

        return eval_node(tree)


class KnowledgeBaseSearch(BaseTool):
    name = "knowledge_search"
    description = "在本地知识库中检索与问题相关的资料片段，返回带编号的原文。"
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "检索关键词或问题"}},
        "required": ["query"],
    }

    def __init__(self, retriever: HybridRetriever, top_k: int = 3) -> None:
        self._retriever = retriever
        self._top_k = top_k

    def run(self, query: str) -> str:
        hits = self._retriever.retrieve(query, self._top_k)
        if not hits:
            return "知识库中没有找到相关资料。"
        return build_numbered_context(hits, self._top_k)
