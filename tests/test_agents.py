"""
Unit tests for AgentFlow core components.
Run: python -m pytest tests/ -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from agentflow.tools.base import ToolRegistry
from agentflow.tools.calculator import CalculatorTool
from agentflow.tools.python_executor import PythonExecutorTool
from agentflow.rag.chunker import TextChunker
from agentflow.memory.buffer import ShortTermMemory


class TestToolRegistry:
    def test_register_tool(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        assert "calculator" in registry.list_tools()

    def test_get_schemas(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        schemas = registry.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "calculator"

    def test_get_handlers(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        handlers = registry.get_handlers()
        assert "calculator" in handlers
        assert callable(handlers["calculator"])


class TestCalculatorTool:
    def test_simple_expression(self):
        tool = CalculatorTool()
        result = tool.execute(expression="2 + 3 * 4")
        assert result["result"] == 14
        assert result["error"] is None

    def test_math_functions(self):
        tool = CalculatorTool()
        result = tool.execute(expression="sqrt(16) + abs(-3)")
        assert result["result"] == 7

    def test_invalid_expression(self):
        tool = CalculatorTool()
        result = tool.execute(expression="__import__('os').system('dir')")
        assert result["error"] is not None


class TestPythonExecutor:
    def test_basic_execution(self):
        tool = PythonExecutorTool()
        result = tool.execute(code="print('hello world')")
        assert result["success"]
        assert "hello world" in result["output"]

    def test_computation(self):
        tool = PythonExecutorTool()
        result = tool.execute(code="import math; print(math.sqrt(144))")
        assert result["success"]
        assert "12.0" in result["output"]

    def test_sandbox(self):
        tool = PythonExecutorTool()
        result = tool.execute(code="open('/etc/passwd')")
        assert not result["success"]


class TestTextChunker:
    def test_chunk_document(self):
        chunker = TextChunker(chunk_size=200, chunk_overlap=20)
        doc = {"content": "Short. " * 100, "source": "test.txt", "filename": "test.txt"}
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1
        for chunk in chunks:
            assert "content" in chunk
            assert "metadata" in chunk
            assert chunk["metadata"]["filename"] == "test.txt"


class TestShortTermMemory:
    def test_add_and_retrieve(self):
        mem = ShortTermMemory(max_turns=4)
        mem.add("user", "Hello")
        mem.add("assistant", "Hi there!")
        history = mem.get_history()
        assert len(history) == 2
        assert history[0]["content"] == "Hello"

    def test_max_turns_limit(self):
        mem = ShortTermMemory(max_turns=2)
        for i in range(10):
            mem.add("user", f"msg {i}")
        assert len(mem) <= 4  # 2 turns = 4 messages max

    def test_clear(self):
        mem = ShortTermMemory(max_turns=4)
        mem.add("user", "test")
        mem.clear()
        assert len(mem) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
