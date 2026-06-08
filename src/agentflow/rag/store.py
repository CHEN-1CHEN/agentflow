"""
Vector Store — ChromaDB-backed vector storage and similarity search.
"""

import os
from typing import Any

import numpy as np


class VectorStore:
    """Vector database abstraction supporting ChromaDB and FAISS backends."""

    def __init__(self, backend: str = "chromadb", persist_dir: str = "./data/vector_store"):
        self.backend = backend
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None
        self._faiss_index = None
        self._faiss_chunks = []

    def _get_chroma(self):
        if self._client is None:
            import chromadb
            os.makedirs(self.persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._client.get_or_create_collection(name="agentflow_docs")
        return self._collection

    def add(self, chunks: list[dict[str, Any]]) -> int:
        """Add embedded chunks to the vector store. Returns count added."""
        if not chunks:
            return 0

        if self.backend == "chromadb":
            return self._add_chroma(chunks)
        elif self.backend == "faiss":
            return self._add_faiss(chunks)
        return 0

    def _add_chroma(self, chunks: list[dict]) -> int:
        collection = self._get_chroma()
        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            source = chunk["metadata"]["source"]
            chunk_idx = chunk["metadata"]["chunk_index"]
            ids.append(f"{source}__chunk_{chunk_idx}")
            documents.append(chunk["content"])
            embeddings.append(chunk["embedding"].tolist())
            metadatas.append(chunk["metadata"])

        # Avoid duplicate IDs
        existing = collection.get(ids=ids)
        if existing and existing["ids"]:
            collection.delete(ids=ids)

        collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        return len(ids)

    def _add_faiss(self, chunks: list[dict]) -> int:
        try:
            import faiss
            vectors = np.array([c["embedding"] for c in chunks], dtype=np.float32)
            if self._faiss_index is None:
                self._faiss_index = faiss.IndexFlatIP(vectors.shape[1])
            self._faiss_index.add(vectors)
            self._faiss_chunks.extend(chunks)
            return len(chunks)
        except ImportError:
            return 0

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        """Search for most similar chunks given a query embedding."""
        if self.backend == "chromadb":
            return self._search_chroma(query_embedding, top_k)
        elif self.backend == "faiss":
            return self._search_faiss(query_embedding, top_k)
        return []

    def _search_chroma(self, query_embedding: np.ndarray, top_k: int) -> list[dict]:
        collection = self._get_chroma()
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )
        output = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                output.append({
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": results["distances"][0][i] if results.get("distances") else 0.0,
                })
        return output

    def _search_faiss(self, query_embedding: np.ndarray, top_k: int) -> list[dict]:
        if self._faiss_index is None:
            return []
        query = np.array([query_embedding], dtype=np.float32)
        scores, indices = self._faiss_index.search(query, top_k)
        return [
            {
                "content": self._faiss_chunks[idx]["content"],
                "metadata": self._faiss_chunks[idx]["metadata"],
                "score": float(scores[0][i]),
            }
            for i, idx in enumerate(indices[0])
            if idx >= 0 and idx < len(self._faiss_chunks)
        ]

    def count(self) -> int:
        if self.backend == "chromadb":
            collection = self._get_chroma()
            return collection.count()
        return len(self._faiss_chunks)
