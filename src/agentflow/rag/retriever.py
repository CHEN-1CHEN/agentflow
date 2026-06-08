"""
Retriever 鈥?high-level RAG retrieval interface combining embedder + vector store.
"""

from typing import Any

from .embedder import Embedder
from .store import VectorStore


class Retriever:
    """High-level retrieval interface for RAG queries."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore, top_k: int = 5):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Retrieve relevant document chunks for a query string."""
        k = top_k or self.top_k
        query_embedding = self.embedder.embed(query)
        return self.vector_store.search(query_embedding, top_k=k)

    def format_context(self, results: list[dict[str, Any]], max_tokens: int = 3000) -> str:
        """Format retrieval results into a context string for LLM prompt injection."""
        if not results:
            return ""

        parts = []
        total_chars = 0
        char_limit = max_tokens * 3  # rough char estimate

        for i, r in enumerate(results):
            snippet = f"[Source {i+1}: {r['metadata'].get('filename', 'unknown')}]\n{r['content']}"
            if total_chars + len(snippet) > char_limit:
                break
            parts.append(snippet)
            total_chars += len(snippet)

        return "\n\n---\n".join(parts)
