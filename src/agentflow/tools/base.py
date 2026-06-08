"""
Tool Registry and Base Tool 鈥?Function Calling infrastructure for Agent tool use.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Callable

from rich.console import Console


class BaseTool(ABC):
    """Abstract tool with OpenAI function-calling schema support."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def get_parameters(self) -> dict:
        """Return JSON Schema for tool parameters."""
        ...

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        ...

    def get_schema(self) -> dict:
        """Generate OpenAI function-calling compatible schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_parameters(),
            },
        }

    def __call__(self, **kwargs) -> Any:
        return self.execute(**kwargs)


class ToolRegistry:
    """Registry for managing, registering, and dispatching tools."""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self.console = Console()

    def register(self, tool: BaseTool) -> "ToolRegistry":
        self._tools[tool.name] = tool
        return self

    def get_schemas(self) -> list[dict]:
        return [t.get_schema() for t in self._tools.values()]

    def get_handlers(self) -> dict[str, Callable]:
        return {name: tool.execute for name, tool in self._tools.items()}

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def __repr__(self) -> str:
        tools = "\n".join(f"  鈥?{n}: {t.description[:60]}" for n, t in self._tools.items())
        return f"ToolRegistry({len(self._tools)} tools):\n{tools}"
