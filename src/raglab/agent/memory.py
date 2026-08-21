"""简单的滑动窗口对话记忆。"""

from __future__ import annotations


class SlidingWindowMemory:
    def __init__(self, max_messages: int = 12) -> None:
        self._messages: list[dict[str, str]] = []
        self.max_messages = max_messages

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages :]

    def messages(self) -> list[dict[str, str]]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()
