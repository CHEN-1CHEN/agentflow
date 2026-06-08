"""
Demo 1: Multi-Agent Task Orchestration
======================================
Demonstrates Planner → Executor → Reviewer workflow for complex task solving.
Run: python examples/demo_multi_agent.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentflow.core import LLMClient, Orchestrator
from agentflow.tools import ToolRegistry, CalculatorTool, PythonExecutorTool, SearchTool, FileOpsTool


def main():
    # Setup LLM client (supports OpenAI / Ollama / domestic APIs)
    llm = LLMClient(config_path=os.path.join(os.path.dirname(__file__), "..", "config.yaml"))

    # Register tools
    tools = ToolRegistry()
    tools.register(CalculatorTool())
    tools.register(PythonExecutorTool(timeout=30))
    tools.register(SearchTool())
    tools.register(FileOpsTool(workspace="."))
    print(f"Tools loaded: {tools.list_tools()}")

    # Create orchestrator
    orchestrator = Orchestrator(llm=llm, tools=tools, max_retries=2, verbose=True)

    # Example tasks demonstrating different Agent capabilities
    tasks = [
        """Analyze the following dataset and provide insights:
        Monthly sales data (Q1 2025):
        Jan: 120 units, $24,000 revenue
        Feb: 145 units, $29,000 revenue
        Mar: 168 units, $33,600 revenue
        Calculate: total revenue, average growth rate, and project Q2 revenue.""",

        """Research task: Compare Transformer, Mamba, and RWKV architectures
        in terms of computational complexity, context length handling, and
        practical performance. Provide a structured comparison table.""",
    ]

    for i, task in enumerate(tasks, 1):
        print(f"\n\n{'#'*60}")
        print(f"# DEMO TASK {i}")
        print(f"{'#'*60}")

        result = orchestrator.run(task)

        print(f"\n--- Task {i} Summary ---")
        print(result["summary"])

    print("\nAll demos complete.")


if __name__ == "__main__":
    main()
