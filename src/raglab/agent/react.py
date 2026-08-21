"""ReAct Agent：思考 → 行动 → 观察 循环，最多迭代 N 次后给出最终答案。"""

from __future__ import annotations

import json
import re

from raglab.agent.memory import SlidingWindowMemory
from raglab.agent.tools import BaseTool
from raglab.llm import BaseLLM

_FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.S)
_ACTION_RE = re.compile(r"Action:\s*(\w+)")
_INPUT_RE = re.compile(r"Action Input:\s*(\{.*?\}|.+?)(?=\nThought:|$)", re.S)

REACT_SYSTEM_PROMPT = """你是一个能够调用工具解决问题的助手。请严格按以下格式回答：

Thought: 你的思考
Action: 工具名
Action Input: {{"参数名": "参数值"}}
Observation: （系统返回，不要自己写）

得到足够信息后，用以下格式结束：
Thought: 我已有足够信息
Final Answer: 最终答案

可用工具：
{tool_specs}"""


class ReActAgent:
    def __init__(
        self,
        llm: BaseLLM,
        tools: list[BaseTool],
        max_iterations: int = 5,
        memory: SlidingWindowMemory | None = None,
    ) -> None:
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.memory = memory or SlidingWindowMemory()
        self.trace: list[dict] = []

    def run(self, question: str) -> str:
        self.trace = []
        self.memory.add("user", question)
        messages = [
            {
                "role": "system",
                "content": REACT_SYSTEM_PROMPT.format(
                    tool_specs=json.dumps(
                        [t.to_spec() for t in self.tools.values()], ensure_ascii=False
                    )
                ),
            },
            *self.memory.messages(),
        ]

        for step in range(self.max_iterations):
            result = self.llm.complete(messages, temperature=0.0)
            output = result.text.strip()
            self.trace.append({"step": step + 1, "output": output})

            final = _FINAL_RE.search(output)
            if final:
                self.memory.add("assistant", output)
                return final.group(1).strip()

            action = _ACTION_RE.search(output)
            action_input = _INPUT_RE.search(output)
            if not action:
                self.memory.add("assistant", output)
                return output

            tool = self.tools.get(action.group(1))
            if tool is None:
                observation = f"错误：未知工具 {action.group(1)}"
            else:
                try:
                    payload = json.loads(action_input.group(1)) if action_input else {}
                    observation = tool.run(**payload)
                except Exception as exc:  # noqa: BLE001
                    observation = f"工具执行失败：{exc}"

            self.trace[-1]["observation"] = observation
            messages.append({"role": "assistant", "content": output})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        self.memory.add("assistant", "达到最大迭代次数，未能完成。")
        return "达到最大迭代次数，未能完成。"
