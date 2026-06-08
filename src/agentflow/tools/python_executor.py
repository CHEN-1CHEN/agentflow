"""
Python Executor Tool 鈥?sandboxed execution of Python code snippets.
"""

import io
import sys
import traceback
from typing import Any

from .base import BaseTool


class PythonExecutorTool(BaseTool):
    """Executes Python code in a restricted sandbox environment."""

    def __init__(self, timeout: int = 30):
        super().__init__(
            name="python_executor",
            description="Execute Python code in a sandbox. Useful for data analysis, calculations, and automation. Returns stdout output.",
        )
        self.timeout = timeout

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Use print() to output results.",
                }
            },
            "required": ["code"],
        }

    def execute(self, code: str, **kwargs) -> Any:
        old_stdout = sys.stdout
        sys.stdout = captured = io.StringIO()

        safe_globals = {
            "__builtins__": {
                "print": print, "len": len, "range": range,
                "int": int, "float": float, "str": str, "bool": bool,
                "list": list, "dict": dict, "set": set, "tuple": tuple,
                "sum": sum, "min": min, "max": max, "sorted": sorted,
                "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
                "abs": abs, "round": round, "isinstance": isinstance,
                "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
            },
        }

        try:
            exec(code, safe_globals, {})
            output = captured.getvalue()
            return {"success": True, "output": output.strip() or "(no output)", "error": None}
        except Exception:
            return {"success": False, "output": captured.getvalue().strip(), "error": traceback.format_exc()}
        finally:
            sys.stdout = old_stdout
