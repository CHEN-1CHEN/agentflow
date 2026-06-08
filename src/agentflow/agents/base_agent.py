"""
Base Agent class with tool registry, memory, and message construction.
"""

from abc import ABC, abstractmethod
from typing import Any

from ..core.llm_client import LLMClient
from ..memory.buffer import ShortTermMemory
from ..tools.base import ToolRegistry


class BaseAgent(ABC):
    """Abstract base for all agents in the system."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm: LLMClient,
        tools: ToolRegistry | None = None,
        memory: ShortTermMemory | None = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.memory = memory or ShortTermMemory()

    def _build_messages(self, user_input: str, include_history: bool = True) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        if include_history:
            for turn in self.memory.get_history():
                messages.append(turn)
        messages.append({"role": "user", "content": user_input})
        return messages

    def _build_tool_schemas(self) -> list[dict] | None:
        schemas = self.tools.get_schemas()
        return schemas if schemas else None

    @abstractmethod
    def run(self, task: str, **kwargs) -> dict[str, Any]:
        """Execute the agent's primary function."""
        ...
