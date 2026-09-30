from rag.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from rag.llms.ollama_llm import OllamaLLM
from rag.memory.memory_extractor import MemoryExtractor
from rag.memory.memory_manager import MemoryManager
from rag.memory.models import MemoryItem
from rag.memory.vector_memory import VectorMemory
from rag.vectorstores.faiss_vector_store import FAISSVectorStore
from pathlib import Path

def create_memory_manager():

    llm = OllamaLLM(
        model="qwen3:8b"
    )

    extractor = MemoryExtractor(
        llm=llm
    )

    embedder = SentenceTransformerEmbedder(
        "BAAI/bge-small-en-v1.5"
    )

    vector_store = FAISSVectorStore(
        dimension=embedder.dimension
    )

    vector_memory = VectorMemory(
        embedder=embedder,
        vector_store=vector_store,
    )

    return MemoryManager(
        extractor=extractor,
        vector_memory=vector_memory,
        index_path=Path("data/memory/test.index"),
        metadata_path=Path("data/memory/test.pkl"),
    )


def test_recall_relevant():

    memory_manager = create_memory_manager()

    memory_manager.vector_memory.add_many(
        [
            MemoryItem(
                id=None,
                content=(
                    "User's favorite programming "
                    "language is Python."
                ),
                memory_type="preference",
            ),
            MemoryItem(
                id=None,
                content=(
                    "User is building a RAG agent "
                    "in Python."
                ),
                memory_type="project",
            ),
            MemoryItem(
                id=None,
                content="User likes basketball.",
                memory_type="interest",
            ),
            MemoryItem(
                id=None,
                content=(
                    "User wants to build a "
                    "Jarvis-like AI assistant."
                ),
                memory_type="goal",
            ),
        ]
    )

    # ==========================================
    # Relevant
    # ==========================================

    memories = memory_manager.recall_relevant(
        "What is my favorite programming language?",
        k=4,
    )

    assert len(memories) >= 1

    assert any(
        "favorite programming language is Python"
        in memory.content
        for memory in memories
    )

    # ==========================================
    # Irrelevant
    # ==========================================

    memories = memory_manager.recall_relevant(
        "What is my GitHub?",
        k=4,
    )

    assert memories == []

    # ==========================================
    # Another relevant memory
    # ==========================================

    memories = memory_manager.recall_relevant(
        "What sport do I like?",
        k=4,
    )

    assert len(memories) >= 1

    assert any(
        "basketball" in memory.content
        for memory in memories
    )

    # ==========================================
    # Different attribute
    # ==========================================

    memories = memory_manager.recall_relevant(
        "What is my favorite programming language?",
        k=4,
    )

    assert not any(
        memory.memory_type == "project"
        for memory in memories
    )