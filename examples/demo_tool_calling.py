"""
Demo 3: Function Calling / Tool Use
====================================
Demonstrates LLM-driven tool calling with automatic dispatch and result handling.
Run: python examples/demo_tool_calling.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentflow.core import LLMClient
from agentflow.tools import ToolRegistry, CalculatorTool, PythonExecutorTool, SearchTool


def main():
    llm = LLMClient(config_path=os.path.join(os.path.dirname(__file__), "..", "config.yaml"))

    # Register tools
    tools = ToolRegistry()
    tools.register(CalculatorTool())
    tools.register(PythonExecutorTool(timeout=30))
    tools.register(SearchTool())
    print(f"Registered tools: {tools.list_tools()}\n")

    # Test 1: Calculator tool
    print("=" * 50)
    print("TEST 1: Calculator Tool")
    print("=" * 50)
    response = llm.chat_with_tools(
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use tools when appropriate."},
            {"role": "user", "content": "Calculate: sqrt(256) + (15 * 3.14) - 2^5"},
        ],
        tools=tools.get_schemas(),
        tool_handlers=tools.get_handlers(),
    )
    print(f"Answer: {response['answer']}")
    print(f"Tool calls made: {len(response['tool_calls'])}")
    for tc in response["tool_calls"]:
        print(f"  鈥?{tc['tool']}({tc['args']}) 鈫?{tc['result'][:100]}")

    # Test 2: Python executor
    print("\n" + "=" * 50)
    print("TEST 2: Python Executor Tool")
    print("=" * 50)
    response = llm.chat_with_tools(
        messages=[
            {"role": "system", "content": "You are a data analyst assistant. Use Python to analyze data."},
            {"role": "user", "content": """Using Python, analyze this data:
            temperatures = [23, 25, 21, 28, 30, 26, 24, 27]
            Calculate: mean, median, standard deviation, and find the max temperature day.""",
            },
        ],
        tools=tools.get_schemas(),
        tool_handlers=tools.get_handlers(),
    )
    print(f"Answer: {response['answer']}")
    print(f"Tool calls made: {len(response['tool_calls'])}")

    # Test 3: Multi-tool chain
    print("\n" + "=" * 50)
    print("TEST 3: Multi-Tool Chain")
    print("=" * 50)
    response = llm.chat_with_tools(
        messages=[
            {"role": "system", "content": "You are a research assistant. Use tools to gather and analyze information."},
            {"role": "user", "content": "Search for the latest UN population projection for 2050, then calculate what annual growth rate that implies from the current 8 billion population."},
        ],
        tools=tools.get_schemas(),
        tool_handlers=tools.get_handlers(),
        max_rounds=5,
    )
    print(f"Answer: {response['answer']}")
    print(f"Tool calls made: {len(response['tool_calls'])}")
    for tc in response["tool_calls"]:
        print(f"  鈥?{tc['tool']}({tc['args']})")

    print("\nAll tool calling tests complete.")


if __name__ == "__main__":
    main()
