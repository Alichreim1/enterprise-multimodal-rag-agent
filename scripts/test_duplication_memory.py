from pathlib import Path

from rag.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from rag.llms.ollama_llm import OllamaLLM
from rag.memory.memory_extractor import MemoryExtractor
from rag.memory.memory_manager import MemoryManager
from rag.memory.vector_memory import VectorMemory
from rag.vectorstores.faiss_vector_store import FAISSVectorStore


def main():

    print("\n===== MEMORY DEDUPLICATION TEST =====\n")

    # ==================================================
    # Embedder
    # ==================================================

    embedder = SentenceTransformerEmbedder()

    # ==================================================
    # Vector store
    # ==================================================

    vector_store = FAISSVectorStore(
        dimension=embedder.dimension
    )

    # ==================================================
    # Vector memory
    # ==================================================

    vector_memory = VectorMemory(
        embedder=embedder,
        vector_store=vector_store,
    )

    # ==================================================
    # LLM
    # ==================================================

    llm = OllamaLLM(
    model="qwen2.5:3b"
)

    # ==================================================
    # Memory extractor
    # ==================================================

    extractor = MemoryExtractor(
        llm=llm
    )

    # ==================================================
    # Temporary paths
    # ==================================================

    index_path = Path(
        "data/test_memory.index"
    )

    metadata_path = Path(
        "data/test_memory.pkl"
    )

    # ==================================================
    # Memory manager
    # ==================================================

    memory_manager = MemoryManager(
        extractor=extractor,
        vector_memory=vector_memory,
        index_path=index_path,
        metadata_path=metadata_path,
    )

    # ==================================================
    # Test inputs
    # ==================================================

    texts = [
        "I prefer Python.",
        "I prefer Python",
        "My preferred programming language is Python.",
        "I like basketball.",
        "I prefer Java.",
    ]

    # ==================================================
    # Remember each input
    # ==================================================

    for text in texts:

        print(
            f"\nUSER: {text}"
        )

        added = memory_manager.remember(
            text
        )

        if added:

            for memory in added:

                print(
                    f"[STORED] {memory.content}"
                )

        else:

            print(
                "[NOT STORED]"
            )

    # ==================================================
    # Final state
    # ==================================================

    print(
        "\n===== FINAL MEMORIES ====="
    )

    memories = memory_manager.recall(
        query="user preferences and interests",
        k=10,
    )

    for memory in memories:

        print(
            f"- {memory.content}"
        )

    print(
        f"\nMemory count: "
        f"{memory_manager.size}"
    )

    # ==================================================
    # Cleanup
    # ==================================================

    memory_manager.clear()


if __name__ == "__main__":
    main()