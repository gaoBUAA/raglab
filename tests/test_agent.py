from raglab.agent import Calculator, ReActAgent
from raglab.llm import MockLLM


def test_calculator_safe_eval():
    tool = Calculator()
    assert tool.run("(2+3)*4") == "(2+3)*4 = 20"


def test_calculator_rejects_code():
    tool = Calculator()
    try:
        tool.run("__import__('os').system('echo hi')")
    except Exception:
        return
    raise AssertionError("危险表达式未被拒绝")


def test_react_agent_reaches_final_answer():
    script = [
        'Thought: 需要计算\nAction: calculator\nAction Input: {"expression": "2*21"}',
        "Thought: 计算完成\nFinal Answer: 42",
    ]

    class ScriptedLLM(MockLLM):
        def __init__(self):
            self.calls = 0

        def complete(self, messages, temperature=None, max_tokens=None):
            text = script[min(self.calls, len(script) - 1)]
            self.calls += 1
            from raglab.llm.base import LLMResult

            return LLMResult(text=text)

    agent = ReActAgent(llm=ScriptedLLM(), tools=[Calculator()], max_iterations=3)
    assert agent.run("21*2 等于多少？") == "42"
