"""
Calculator Tool 鈥?safe mathematical expression evaluation.
"""

import math
import operator
from typing import Any

from .base import BaseTool


class CalculatorTool(BaseTool):
    """Evaluates mathematical expressions safely using restricted builtins."""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="Evaluate a mathematical expression. Supports: +, -, *, /, **, %, sqrt, sin, cos, log, abs, round, pi, e.",
        )

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate, e.g. 'sqrt(16) + 3 * 2'",
                }
            },
            "required": ["expression"],
        }

    def execute(self, expression: str, **kwargs) -> Any:
        allowed = {
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "log": math.log, "log10": math.log10, "abs": abs, "round": round,
            "pi": math.pi, "e": math.e, "pow": pow, "ceil": math.ceil, "floor": math.floor,
        }
        try:
            result = eval(expression, {"__builtins__": {}}, allowed)
            return {"expression": expression, "result": result, "error": None}
        except Exception as e:
            return {"expression": expression, "result": None, "error": str(e)}
