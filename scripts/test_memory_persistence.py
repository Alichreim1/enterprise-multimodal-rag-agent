from pathlib import Path

from rag.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from rag.memory.models import MemoryItem
from rag.memory.vector_memory import VectorMemory
from rag.vectorstores.faiss_vector_store import FAISSVectorStore


MEMORY_DIR = Path("data/memory")

INDEX_PATH = MEMORY_DIR / "memories.index"
METADATA_PATH = MEMORY_DIR / "memories.pkl"


# ==================================================
# Create components
# ==================================================

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

memories = [
    MemoryItem(
        id="1",
        content="User's favorite programming language is Python.",
        memory_type="preference",
    ),
    MemoryItem(
        id="2",
        content="User is building a RAG agent in Python.",
        memory_type="project",
    ),
    MemoryItem(
        id="3",
        content="User likes basketball.",
        memory_type="interest",
    ),
    MemoryItem(
        id="4",
        content="User wants to build a Jarvis-like AI assistant.",
        memory_type="goal",
    ),
]


memory.add_many(memories)


print("\nBEFORE SAVE")
print("Memory size:", memory.size)


# ==================================================
# Save
# ==================================================

memory.save(
    index_path=INDEX_PATH,
    metadata_path=METADATA_PATH,
)


print("\nFILES CREATED")

print(
    "Index:",
    INDEX_PATH,
    "->",
    INDEX_PATH.exists(),
)

print(
    "Metadata:",
    METADATA_PATH,
    "->",
    METADATA_PATH.exists(),
)


# ==================================================
# Create completely new memory instance
# ==================================================

new_vector_store = FAISSVectorStore(
    dimension=embedder.dimension
)

new_memory = VectorMemory(
    embedder=embedder,
    vector_store=new_vector_store,
)


# ==================================================
# Load
# ==================================================

new_memory.load(
    index_path=INDEX_PATH,
    metadata_path=METADATA_PATH,
)


print("\nAFTER LOAD")
print("Memory size:", new_memory.size)


# ==================================================
# Retrieval after loading
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

    results = new_memory.retrieve(
        query,
        k=1,
    )

    print("\nRESULT:")

    for result in results:

        print(
            result.memory_type,
            "->",
            result.content,
        )