"""
Executor Agent — executes individual plan steps with tool calling capability.
Uses ReAct (Reasoning + Acting) pattern for dynamic tool selection and execution.
"""

import json
from typing import Any

from .base_agent import BaseAgent
from ..tools.base import ToolRegistry

EXECUTOR_SYSTEM_PROMPT = """You are an execution agent. Given a specific sub-task, you must complete it using available tools and reasoning.

## Execution Protocol (ReAct Pattern)
1. **Thought**: Analyze what needs to be done
2. **Action**: Call the appropriate tool with correct parameters (if needed)
3. **Observation**: Review the tool output
4. **Repeat** until the sub-task is fully resolved

## Guidelines
- Use tools whenever a task requires computation, search, or external data
- Verify tool outputs before treating them as final
- If a tool fails, try an alternative approach
- When done, provide a clear, structured final answer

## Output Format
When you have the final answer, wrap it in:
```json
{"status": "success", "result": "your final answer here"}
```
If you cannot complete the task:
```json
{"status": "failed", "reason": "explanation"}
```"""


class ExecutorAgent(BaseAgent):
    """Executes individual steps with tool calling via ReAct pattern."""

    def __init__(self, llm, tools: ToolRegistry, memory=None):
        super().__init__(
            name="Executor",
            system_prompt=EXECUTOR_SYSTEM_PROMPT,
            llm=llm,
            tools=tools,
            memory=memory,
        )

    def run(self, task: str, context: dict | None = None, **kwargs) -> dict[str, Any]:
        user_input = task
        if context:
            user_input = f"Context from previous steps:\n{json.dumps(context, ensure_ascii=False)}\n\nCurrent task:\n{task}"

        messages = self._build_messages(user_input)
        tool_schemas = self._build_tool_schemas()

        if tool_schemas:
            response = self.llm.chat_with_tools(
                messages=messages,
                tools=tool_schemas,
                tool_handlers=self.tools.get_handlers(),
                max_rounds=8,
            )
            result_text = response.get("answer", "")
            tool_calls = response.get("tool_calls", [])
        else:
            response = self.llm.chat(messages=messages)
            result_text = response.get("content", "")
            tool_calls = []

        try:
            if "```json" in result_text:
                json_str = result_text.split("```json")[1].split("```")[0].strip()
                parsed = json.loads(json_str)
            elif result_text.strip().startswith("{"):
                parsed = json.loads(result_text.strip())
            else:
                parsed = {"status": "success", "result": result_text}
        except json.JSONDecodeError:
            parsed = {"status": "success", "result": result_text}

        self.memory.add("user", task)
        self.memory.add("assistant", json.dumps(parsed, ensure_ascii=False))

        return {
            "output": parsed,
            "tool_calls": tool_calls,
            "raw_response": result_text,
        }
