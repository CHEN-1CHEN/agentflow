"""
Orchestrator — coordinates the Planner → Executor → Reviewer loop
for multi-agent task execution with feedback-driven retry.
"""

import json
from typing import Any

from .llm_client import LLMClient
from ..agents.planner import PlannerAgent
from ..agents.executor import ExecutorAgent
from ..agents.reviewer import ReviewerAgent
from ..memory.buffer import ShortTermMemory
from ..tools.base import ToolRegistry


class Orchestrator:
    """Coordinates multi-agent workflow: Plan → Execute → Review → (Retry|Next)."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry | None = None,
        max_retries: int = 3,
        verbose: bool = True,
    ):
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.max_retries = max_retries
        self.verbose = verbose

        self.planner = PlannerAgent(llm=self.llm)
        self.executor = ExecutorAgent(llm=self.llm, tools=self.tools)
        self.reviewer = ReviewerAgent(llm=self.llm)

    def run(self, task: str, context: dict | None = None) -> dict[str, Any]:
        """Execute a full Plan-Execute-Review cycle for a given task."""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  Orchestrator: Processing Task")
            print(f"  Goal: {task[:100]}{'...' if len(task) > 100 else ''}")
            print(f"{'='*60}")

        # Phase 1: Planning
        if self.verbose:
            print("\n[Phase 1] PlannerAgent — Decomposing task...")
        plan = self.planner.run(task)
        steps = plan.get("plan", [])
        if self.verbose:
            print(f"  Complexity: {plan.get('estimated_complexity', 'unknown')}")
            print(f"  Steps: {len(steps)}")
            for s in steps:
                print(f"    {s['step_id']}. {s['description'][:80]}")

        # Phase 2-3: Execute + Review each step
        results = []
        step_context = {}

        for step in steps:
            step_id = step.get("step_id", "?")
            if self.verbose:
                print(f"\n[Phase 2] ExecutorAgent — Step {step_id}: {step['description'][:80]}")

            # Resolve dependencies
            deps = step.get("depends_on", [])
            dep_results = {d: step_context.get(d) for d in deps if d in step_context}

            for retry in range(self.max_retries):
                exec_result = self.executor.run(
                    task=step["description"],
                    context={**context or {}, **dep_results} if (context or dep_results) else None,
                )

                if self.verbose:
                    status = exec_result.get("output", {}).get("status", "?")
                    tc_count = len(exec_result.get("tool_calls", []))
                    print(f"  Status: {status} | Tool calls: {tc_count}")

                # Phase 3: Review
                if self.verbose:
                    print(f"[Phase 3] ReviewerAgent — Validating step {step_id}...")
                review = self.reviewer.review(
                    step=step,
                    execution_result=exec_result,
                    max_retries=1,
                )

                if self.verbose:
                    print(f"  Verdict: {review.get('verdict')} | Score: {review.get('score', '?')}")

                if review.get("verdict") == "pass":
                    break
                elif review.get("verdict") == "fail":
                    if self.verbose:
                        print(f"  FAILED: {review.get('feedback', '')[:100]}")
                    break
                else:
                    if self.verbose:
                        print(f"  Retry {retry + 1}/{self.max_retries}: {review.get('feedback', '')[:80]}")

            step_context[step_id] = exec_result.get("output", {}).get("result", "")
            results.append({
                "step_id": step_id,
                "description": step["description"],
                "result": exec_result.get("output", {}).get("result", ""),
                "status": exec_result.get("output", {}).get("status", "unknown"),
                "review": review,
                "tool_calls": exec_result.get("tool_calls", []),
            })

        if self.verbose:
            passed = sum(1 for r in results if r["review"].get("verdict") == "pass")
            print(f"\n{'='*60}")
            print(f"  Complete: {passed}/{len(results)} steps passed")
            print(f"{'='*60}\n")

        return {
            "goal": task,
            "plan": plan,
            "results": results,
            "summary": self._summarize(results),
        }

    def _summarize(self, results: list[dict]) -> str:
        passed = sum(1 for r in results if r["review"].get("verdict") == "pass")
        failed = sum(1 for r in results if r["review"].get("verdict") == "fail")
        total = len(results)
        avg_score = sum(r["review"].get("score", 0) for r in results) / max(total, 1)
        total_tool_calls = sum(len(r.get("tool_calls", [])) for r in results)
        return json.dumps({
            "total_steps": total,
            "passed": passed,
            "failed": failed,
            "average_score": round(avg_score, 1),
            "total_tool_calls": total_tool_calls,
        }, ensure_ascii=False)
