"""
Embedder — text to vector embedding using sentence-transformers.
"""

import numpy as np
from typing import Any


class Embedder:
    """Generates dense vector embeddings for text chunks."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: str | list[str]) -> np.ndarray:
        """Generate embeddings for one or more texts."""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return embeddings

    def embed_documents(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add embeddings to document chunks."""
        if not chunks:
            return []
        texts = [c["content"] for c in chunks]
        vectors = self.embed(texts)
        for chunk, vector in zip(chunks, vectors):
            chunk["embedding"] = vector
        return chunks
