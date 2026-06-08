"""
Text Chunker 鈥?splits documents into overlapping chunks for RAG indexing.
"""

import re
from typing import Any


class TextChunker:
    """Splits text into semantically meaningful, overlapping chunks."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: dict[str, Any]) -> list[dict[str, Any]]:
        """Split a single document into chunks with metadata."""
        text = document["content"]
        if not text or len(text) < 50:
            return []

        chunks = self._split_text(text)
        return [
            {
                "content": chunk,
                "metadata": {
                    "source": document["source"],
                    "filename": document["filename"],
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                },
            }
            for i, chunk in enumerate(chunks)
        ]

    def chunk_documents(self, documents: list[dict]) -> list[dict]:
        """Split multiple documents into chunks."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

    def _split_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks, respecting sentence boundaries."""
        # Split into sentences (Chinese + English aware)
        sentences = re.split(r'(?<=[銆傦紒锛?!?\n])\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) <= self.chunk_size:
                current += sentence
            else:
                if current:
                    chunks.append(current)
                # If a single sentence exceeds chunk_size, split by characters
                if len(sentence) > self.chunk_size:
                    for i in range(0, len(sentence), self.chunk_size - self.chunk_overlap):
                        chunks.append(sentence[i:i + self.chunk_size])
                else:
                    current = sentence

        if current:
            chunks.append(current)

        return chunks
