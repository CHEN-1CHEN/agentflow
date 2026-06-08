"""
Demo 2: RAG (Retrieval-Augmented Generation) Pipeline
======================================================
Build a knowledge base from documents, then query it with LLM-augmented answers.
Run: python examples/demo_rag.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentflow.core import LLMClient
from agentflow.rag import DocumentLoader, TextChunker, Embedder, VectorStore, Retriever


def create_sample_docs(work_dir: str):
    """Create sample documents for the demo if none exist."""
    os.makedirs(work_dir, exist_ok=True)

    # Sample doc 1: Transformer architecture
    with open(os.path.join(work_dir, "transformer_intro.txt"), "w", encoding="utf-8") as f:
        f.write("""
The Transformer architecture, introduced in the paper "Attention Is All You Need" (Vaswani et al., 2017),
revolutionized deep learning by replacing recurrent neural networks with self-attention mechanisms.

Key components of the Transformer:
1. Multi-Head Self-Attention: Allows the model to attend to different positions of the input sequence simultaneously.
   Each attention head learns different relationship patterns between tokens.

2. Positional Encoding: Since Transformers process tokens in parallel rather than sequentially,
   positional encodings inject information about token positions using sinusoidal functions.

3. Feed-Forward Networks (FFN): Each attention layer is followed by a position-wise feed-forward network,
   typically with a hidden dimension 4x the model dimension. Uses GeLU or ReLU activation.

4. Layer Normalization: Applied before each sub-layer (Pre-LN) or after (Post-LN).
   Pre-LN has become standard as it improves training stability.

5. Residual Connections: Wrap each sub-layer to enable gradient flow through deep networks.

The Transformer uses an encoder-decoder architecture:
- Encoder: N identical layers, each with self-attention + FFN
- Decoder: N identical layers, each with masked self-attention + cross-attention + FFN

Major variants include:
- BERT (Devlin et al., 2019): Encoder-only, bidirectional attention, masked language modeling
- GPT (Radford et al., 2018): Decoder-only, unidirectional (causal) attention, next-token prediction
- T5 (Raffel et al., 2020): Full encoder-decoder, text-to-text framework

Computational complexity: O(n^2 * d) for self-attention where n is sequence length and d is model dimension.
""")

    # Sample doc 2: RAG introduction
    with open(os.path.join(work_dir, "rag_intro.txt"), "w", encoding="utf-8") as f:
        f.write("""
Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation
to produce more accurate and grounded responses from language models.

RAG Pipeline:
1. Document Loading: Import documents from various sources (PDF, web pages, databases)
2. Text Chunking: Split documents into semantically meaningful segments (typically 256-1024 tokens)
3. Embedding: Convert text chunks into dense vector representations using models like:
   - OpenAI text-embedding-ada-002 (1536 dimensions)
   - Sentence-BERT (384-768 dimensions)
   - E5 / BGE models (1024 dimensions)
4. Vector Storage: Store embeddings in a vector database:
   - ChromaDB: Open-source, local-first
   - FAISS: Facebook AI Similarity Search, high-performance
   - Pinecone / Weaviate: Managed cloud solutions
   - Milvus: Distributed vector database for large-scale deployments
5. Retrieval: At query time, embed the question and find the top-K most similar chunks
6. Augmented Generation: Inject retrieved context into the LLM prompt for grounded answer generation

Advanced RAG Techniques:
- HyDE (Hypothetical Document Embeddings): Generate a hypothetical answer first, then use it for retrieval
- Parent Document Retrieval: Index small chunks but retrieve larger parent documents for context
- Self-RAG: The model decides when to retrieve, critique its own outputs, and filter irrelevant context
- Multi-hop RAG: Chain multiple retrieval steps for complex questions requiring reasoning across documents
- RAPTOR: Recursive summarization and tree-based indexing for multi-scale document understanding

Evaluation Metrics for RAG:
- Faithfulness: Does the answer stay true to retrieved documents?
- Answer Relevance: Does it address the query?
- Context Recall: Are all relevant chunks retrieved?
- Context Precision: Are retrieved chunks actually relevant?

Common Challenges:
- Chunk boundary issues breaking coherent information
- Retrieval failure for queries requiring implicit knowledge
- Hallucination when retrieved context is insufficient
- Latency overhead from the retrieval step
""")

    # Sample doc 3: Agent types
    with open(os.path.join(work_dir, "agent_types.txt"), "w", encoding="utf-8") as f:
        f.write("""
AI Agents are autonomous systems that perceive their environment, make decisions, and take actions
to achieve specific goals. Modern LLM-based agents leverage large language models as their reasoning core.

Agent Architecture Patterns:

1. ReAct (Reasoning + Acting): Interleaves thought traces with action steps.
   The agent thinks about what to do, executes an action, observes the result, and repeats.
   This creates an interpretable decision trace.

2. Plan-and-Execute: First creates a complete plan, then executes each step.
   More structured but less flexible than ReAct. Better for tasks with clear sub-goals.

3. Multi-Agent Systems: Multiple specialized agents collaborate:
   - Planner Agent: Task decomposition and strategy
   - Executor Agent: Action execution with tool use
   - Reviewer Agent: Quality validation and feedback
   - Memory Agent: Knowledge storage and retrieval

4. Reflexion: The agent reflects on its own outputs, identifies errors, and improves.
   Uses verbal reinforcement learning — the agent critiques its own reasoning trace.

5. Tool-Augmented Agents (Function Calling): Agents call external tools via structured APIs.
   Tools can include: calculators, search engines, code interpreters, databases, APIs.

6. Tree-of-Thought (ToT): Explores multiple reasoning paths simultaneously,
   evaluates them, and selects the best. Enables backtracking from dead ends.

Key Components of LLM Agents:
- Planning: Breaking complex goals into manageable sub-tasks
- Memory: Short-term (conversation buffer) and long-term (vector store) memory
- Tool Use: Calling external functions to extend capabilities beyond text generation
- Reasoning: Chain-of-thought, self-consistency, and structured reasoning techniques

Evaluation of Agents:
- Task Success Rate (TSR)
- Tool Call Accuracy
- Plan Quality Score
- Execution Efficiency (steps per task)
- Robustness to environmental changes
""")

    print(f"Created 3 sample documents in {work_dir}/")


def main():
    # Config
    docs_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs")
    persist_dir = os.path.join(os.path.dirname(__file__), "..", "data", "vector_store")

    # Create sample docs
    create_sample_docs(docs_dir)

    # Initialize RAG pipeline
    print("\n[1/5] Loading documents...")
    loader = DocumentLoader()
    documents = loader.load(docs_dir)
    print(f"  Loaded {len(documents)} documents")

    print("\n[2/5] Chunking documents...")
    chunker = TextChunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk_documents(documents)
    print(f"  Created {len(chunks)} chunks")

    print("\n[3/5] Generating embeddings...")
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    chunks = embedder.embed_documents(chunks)
    print(f"  Embedded {len(chunks)} chunks (dim={chunks[0]['embedding'].shape[0]})")

    print("\n[4/5] Building vector store...")
    vector_store = VectorStore(backend="chromadb", persist_dir=persist_dir)
    count = vector_store.add(chunks)
    print(f"  Indexed {count} vectors in vector store")

    print("\n[5/5] Running retrieval queries...")
    retriever = Retriever(embedder=embedder, vector_store=vector_store, top_k=3)

    queries = [
        "What is the computational complexity of Transformer self-attention?",
        "How does RAG combine retrieval with generation?",
        "What are the different types of AI agent architectures?",
        "Compare ReAct and Plan-and-Execute agent patterns",
    ]

    for query in queries:
        print(f"\n{'─'*60}")
        print(f"Query: {query}")

        results = retriever.retrieve(query, top_k=3)
        context = retriever.format_context(results)
        print(f"Retrieved {len(results)} relevant chunks")

        # Show top result snippet
        if results:
            top = results[0]
            print(f"  Top match: [{top['metadata']['filename']}] (score: {top['score']:.3f})")
            print(f"  Preview: {top['content'][:150]}...")

    print(f"\n{'='*60}")
    print(f"RAG pipeline demo complete. Vector store: {vector_store.count()} documents indexed.")
    print(f"Data persisted at: {persist_dir}")


if __name__ == "__main__":
    main()
