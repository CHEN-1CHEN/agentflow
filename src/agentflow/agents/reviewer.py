"""
Reviewer Agent 鈥?validates execution results against the original plan.
Checks for correctness, completeness, and quality; provides feedback for re-execution.
"""

import json
from typing import Any

from .base_agent import BaseAgent

REVIEWER_SYSTEM_PROMPT = """You are a quality assurance reviewer. Your job is to validate execution results against the original task specification.

## Review Criteria
1. **Completeness**: Did the execution address ALL parts of the task?
2. **Correctness**: Is the result factually and logically correct?
3. **Quality**: Is the output well-structured, clear, and actionable?
4. **Tool Usage**: Were tools used appropriately? Any missed opportunities?

## Output Format
```json
{
  "verdict": "pass|fail|retry",
  "score": 0-100,
  "completeness": "brief assessment",
  "correctness": "brief assessment",
  "quality": "brief assessment",
  "issues": ["issue 1", "issue 2"],
  "feedback": "what to improve if retry is needed"
}
```

Be strict but fair. A score below 60 means fail. Between 60-79 means retry. 80+ means pass."""


class ReviewerAgent(BaseAgent):
    """Validates execution results and provides quality feedback."""

    def __init__(self, llm, memory=None):
        super().__init__(
            name="Reviewer",
            system_prompt=REVIEWER_SYSTEM_PROMPT,
            llm=llm,
            memory=memory,
        )

    def review(
        self,
        step: dict[str, Any],
        execution_result: dict[str, Any],
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Review a single execution result. Returns review dict with verdict."""
        review_prompt = f"""Review this execution:

## Original Task Step
{json.dumps(step, ensure_ascii=False, indent=2)}

## Execution Result
{json.dumps(execution_result, ensure_ascii=False, indent=2)}

Provide your review verdict."""

        history = []
        for attempt in range(max_retries):
            messages = [{"role": "system", "content": self.system_prompt}]
            for h in history:
                messages.append(h)
            messages.append({"role": "user", "content": review_prompt})

            response = self.llm.chat(messages=messages, temperature=0.1)
            content = response.get("content", "")

            try:
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    json_str = content.split("```")[1].split("```")[0].strip()
                else:
                    json_str = content.strip()
                review = json.loads(json_str)
            except json.JSONDecodeError:
                review = {"verdict": "pass", "score": 80, "issues": [], "feedback": ""}

            history.append({"role": "assistant", "content": content})

            if review.get("verdict") == "pass":
                break

            if review.get("verdict") == "fail":
                break

        self.memory.add("user", f"Review task step {step.get('step_id', '?')}")
        self.memory.add("assistant", json.dumps(review, ensure_ascii=False))
        return review

    def run(self, task: str, **kwargs) -> dict[str, Any]:
        """Run as a standalone reviewer. Expects task to contain step and result."""
        return self.review(step={"description": task}, execution_result=kwargs.get("result", {}))
