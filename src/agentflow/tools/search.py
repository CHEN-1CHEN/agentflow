"""
Search Tool — web search capability via DuckDuckGo (no API key required).
"""

import json
from typing import Any

from .base import BaseTool


class SearchTool(BaseTool):
    """Web search tool using DuckDuckGo's instant answer and text search APIs."""

    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web for current information. Returns top results with title, URL, and snippet.",
        )

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default 5, max 10)",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    def execute(self, query: str, max_results: int = 5, **kwargs) -> Any:
        try:
            from duckduckgo_search import DDGS
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=min(max_results, 10)):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")[:200],
                    })
            return {"query": query, "results": results, "count": len(results)}
        except ImportError:
            return {"query": query, "results": [], "error": "duckduckgo_search not installed. Install with: pip install duckduckgo-search"}
        except Exception as e:
            return {"query": query, "results": [], "error": str(e)}
