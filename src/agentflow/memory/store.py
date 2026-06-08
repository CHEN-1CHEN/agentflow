"""
Long-Term Memory — persistent vector-based memory for cross-session knowledge retention.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


class LongTermMemory:
    """Persistent memory store backed by vector embeddings for semantic retrieval."""

    def __init__(
        self,
        persist_dir: str = "./data/memory_store",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model_name = embedding_model
        self._embedder = None
        self._memories: list[dict] = []
        self._load()

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def store(self, content: str, metadata: dict | None = None, importance: float = 0.5) -> str:
        """Store a memory with vector embedding for later retrieval."""
        embedding = self.embedder.encode([content], normalize_embeddings=True)[0]
        memory = {
            "id": f"mem_{int(time.time() * 1000)}_{len(self._memories)}",
            "content": content,
            "metadata": metadata or {},
            "importance": importance,
            "embedding": embedding.tolist(),
            "created_at": time.time(),
        }
        self._memories.append(memory)
        self._save()
        return memory["id"]

    def retrieve(self, query: str, top_k: int = 5, threshold: float = 0.3) -> list[dict[str, Any]]:
        """Semantically retrieve relevant memories for a query."""
        if not self._memories:
            return []

        query_emb = self.embedder.encode([query], normalize_embeddings=True)[0]
        results = []

        for mem in self._memories:
            similarity = float(np.dot(query_emb, np.array(mem["embedding"])))
            if similarity >= threshold:
                results.append({
                    "id": mem["id"],
                    "content": mem["content"],
                    "metadata": mem["metadata"],
                    "importance": mem["importance"],
                    "score": similarity,
                })

        results.sort(key=lambda x: x["score"] * x["importance"], reverse=True)
        return results[:top_k]

    def forget(self, memory_id: str) -> bool:
        """Delete a specific memory by ID."""
        before = len(self._memories)
        self._memories = [m for m in self._memories if m["id"] != memory_id]
        if len(self._memories) < before:
            self._save()
            return True
        return False

    def clear(self) -> None:
        """Remove all memories."""
        self._memories.clear()
        self._save()

    def _save(self) -> None:
        save_data = []
        for m in self._memories:
            save_data.append({k: v for k, v in m.items() if k != "embedding"})
        (self.persist_dir / "memories.json").write_text(
            json.dumps(save_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load(self) -> None:
        mem_file = self.persist_dir / "memories.json"
        if mem_file.exists():
            data = json.loads(mem_file.read_text(encoding="utf-8"))
            for item in data:
                embedding = self.embedder.encode([item["content"]], normalize_embeddings=True)[0]
                item["embedding"] = embedding.tolist()
                self._memories.append(item)

    def __len__(self) -> int:
        return len(self._memories)

    def __repr__(self) -> str:
        return f"LongTermMemory(memories={len(self._memories)})"
