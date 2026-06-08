"""
Short-Term Memory — sliding-window conversation buffer for Agent context management.
"""

from collections import deque
from typing import Any


class ShortTermMemory:
    """Fixed-size conversation buffer that maintains recent dialogue turns."""

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self._buffer: deque[dict[str, str]] = deque(maxlen=max_turns * 2)
        self._summaries: list[str] = []

    def add(self, role: str, content: str) -> None:
        self._buffer.append({"role": role, "content": content})

    def get_history(self) -> list[dict[str, str]]:
        return list(self._buffer)

    def get_last_n(self, n: int) -> list[dict[str, str]]:
        history = list(self._buffer)
        return history[-n * 2:] if n * 2 < len(history) else history

    def summarize_and_compress(self, llm, keep_last: int = 4) -> str:
        """Summarize older conversation turns to save context window space."""
        history = list(self._buffer)
        if len(history) <= keep_last * 2:
            return ""

        to_summarize = history[:-keep_last * 2]
        recent = history[-keep_last * 2:]

        summary_prompt = "Summarize this conversation history concisely, preserving key facts and decisions:\n\n"
        for msg in to_summarize:
            summary_prompt += f"[{msg['role']}]: {msg['content'][:500]}\n"

        response = llm.chat(
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.2,
            max_tokens=500,
        )
        summary = response.get("content", "")
        self._summaries.append(summary)

        # Replace buffer: summaries as system context + recent messages
        self._buffer.clear()
        for s in self._summaries[-3:]:
            self._buffer.append({"role": "system", "content": f"[Memory Summary]: {s}"})
        for msg in recent:
            self._buffer.append(msg)

        return summary

    def clear(self) -> None:
        self._buffer.clear()
        self._summaries.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"ShortTermMemory(turns={len(self._buffer)//2}/{self.max_turns})"
