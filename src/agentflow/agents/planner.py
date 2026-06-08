"""
Planner Agent — decomposes complex tasks into structured execution plans.
Uses Chain-of-Thought reasoning for step-by-step task decomposition.
"""

import json
from typing import Any

from .base_agent import BaseAgent

PLANNER_SYSTEM_PROMPT = """You are a strategic task planner. Your role is to decompose complex user requests into clear, executable steps.

## Process
1. Analyze the user's goal and identify all requirements
2. Break down the goal into atomic, ordered sub-tasks
3. For each sub-task, specify: the action, expected output, and any dependencies
4. Ensure steps are logically ordered and independently executable

## Output Format
Respond ONLY with valid JSON:
```json
{
  "goal": "restated user goal",
  "reasoning": "brief chain-of-thought analysis",
  "plan": [
    {
      "step_id": 1,
      "description": "what to do",
      "action_type": "reasoning|tool_call|verification",
      "expected_output": "what this step should produce",
      "depends_on": []
    }
  ],
  "estimated_complexity": "low|medium|high"
}
```

## Constraints
- Each step must be atomic (single action)
- Maximum 10 steps per plan
- Identify dependencies between steps
- Prefer calling tools over raw reasoning when external data is needed"""


class PlannerAgent(BaseAgent):
    """Decomposes user goals into structured, executable plans."""

    def __init__(self, llm, memory=None):
        super().__init__(
            name="Planner",
            system_prompt=PLANNER_SYSTEM_PROMPT,
            llm=llm,
            memory=memory,
        )

    def run(self, task: str, **kwargs) -> dict[str, Any]:
        messages = self._build_messages(task)

        response = self.llm.chat(
            messages=messages,
            temperature=0.2,
        )

        content = response.get("content", "")
        try:
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()
            plan = json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            plan = {
                "goal": task,
                "reasoning": "Failed to parse structured plan",
                "plan": [{"step_id": 1, "description": task, "action_type": "reasoning", "expected_output": "complete", "depends_on": []}],
                "estimated_complexity": "medium",
            }

        self.memory.add("user", task)
        self.memory.add("assistant", json.dumps(plan, ensure_ascii=False))
        return plan
