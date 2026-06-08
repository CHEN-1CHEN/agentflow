"""
File Operations Tool 鈥?read, write, and list files within the workspace.
"""

import os
from pathlib import Path
from typing import Any

from .base import BaseTool


class FileOpsTool(BaseTool):
    """Safe file operations: read, write, list, and search files."""

    def __init__(self, workspace: str = "."):
        super().__init__(
            name="file_operations",
            description="Read, write, list, or search files in the workspace. Use for document processing and data handling.",
        )
        self.workspace = Path(workspace).resolve()

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read", "write", "list", "search"],
                    "description": "The file operation to perform",
                },
                "path": {
                    "type": "string",
                    "description": "File path relative to workspace",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (only for write action)",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern for filenames (only for search action, e.g. '*.py')",
                },
            },
            "required": ["action", "path"],
        }

    def execute(self, action: str, path: str, content: str = "", pattern: str = "", **kwargs) -> Any:
        try:
            target = (self.workspace / path).resolve()
            if not str(target).startswith(str(self.workspace)):
                return {"error": "Path traversal denied"}

            if action == "read":
                if not target.exists():
                    return {"error": f"File not found: {path}"}
                text = target.read_text(encoding="utf-8")
                return {"path": path, "content": text[:5000], "size": len(text)}

            elif action == "write":
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                return {"path": path, "written": len(content), "status": "ok"}

            elif action == "list":
                target_dir = target if target.is_dir() else target.parent
                entries = []
                for item in target_dir.iterdir():
                    entries.append({
                        "name": item.name,
                        "type": "dir" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else 0,
                    })
                return {"path": str(target_dir.relative_to(self.workspace)), "entries": entries[:100]}

            elif action == "search":
                target_dir = target if target.is_dir() else target.parent
                matches = list(target_dir.rglob(pattern or "*"))
                return {"pattern": pattern, "matches": [str(m.relative_to(self.workspace)) for m in matches[:50]]}

            return {"error": f"Unknown action: {action}"}

        except Exception as e:
            return {"error": str(e)}
