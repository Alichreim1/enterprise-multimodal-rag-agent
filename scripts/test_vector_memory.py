from rag.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from rag.memory.models import MemoryItem
from rag.memory.vector_memory import VectorMemory
from rag.vectorstores.faiss_vector_store import FAISSVectorStore


embedder = SentenceTransformerEmbedder(
    "BAAI/bge-small-en-v1.5"
)

vector_store = FAISSVectorStore(
    dimension=embedder.dimension
)

memory = VectorMemory(
    embedder=embedder,
    vector_store=vector_store,
)


# ==================================================
# Add memories
# ==================================================

memory.add_many(
    [
        MemoryItem(
            id=None,
            content="User's favorite programming language is Python.",
            memory_type="preference",
        ),
        MemoryItem(
            id=None,
            content="User is building a RAG agent in Python.",
            memory_type="project",
        ),
        MemoryItem(
            id=None,
            content="User likes basketball.",
            memory_type="interest",
        ),
        MemoryItem(
            id=None,
            content="User wants to build a Jarvis-like AI assistant.",
            memory_type="goal",
        ),
    ]
)


print("\nMEMORY SIZE:")
print(memory.size)


# ==================================================
# Retrieval tests
# ==================================================

queries = [
    "What programming language does the user prefer?",
    "What project is the user working on?",
    "What sport does the user like?",
    "What does the user want to build?",
]


for query in queries:

    print("\n==============================")
    print("QUERY:")
    print(query)

    results = memory.retrieve(
        query,
        k=2,
    )

    print("\nRESULTS:")

    for result in results:

        print(
            result.memory_type,
            "->",
            result.content,
        )