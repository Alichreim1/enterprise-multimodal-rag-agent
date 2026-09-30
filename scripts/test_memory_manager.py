from pathlib import Path

from rag.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from rag.llms.ollama_llm import OllamaLLM
from rag.memory.memory_extractor import MemoryExtractor
from rag.memory.memory_manager import MemoryManager
from rag.memory.vector_memory import VectorMemory
from rag.vectorstores.faiss_vector_store import FAISSVectorStore


# ==================================================
# Paths
# ==================================================

MEMORY_DIR = Path("data/memory")

INDEX_PATH = MEMORY_DIR / "memories.index"
METADATA_PATH = MEMORY_DIR / "memories.pkl"


# ==================================================
# Components
# ==================================================

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

memory_manager = MemoryManager(
    extractor=extractor,
    vector_memory=vector_memory,
    index_path=INDEX_PATH,
    metadata_path=METADATA_PATH,
)


# ==================================================
# Start clean
# ==================================================

memory_manager.clear()


# ==================================================
# Remember
# ==================================================

messages = [
    "My favorite programming language is Python.",
    "I am building a RAG agent in Python.",
    "I like basketball.",
    "I want to build a Jarvis-like AI assistant.",
    "What is 25 * 8?",
]


for message in messages:

    print("\n==============================")
    print("USER:")
    print(message)

    memories = memory_manager.remember(
        message
    )

    print("\nEXTRACTED MEMORIES:")

    for memory in memories:

        print(
            memory.memory_type,
            "->",
            memory.content,
        )


# ==================================================
# Size
# ==================================================

print("\n==============================")
print("MEMORY SIZE:")
print(memory_manager.size)


# ==================================================
# Recall
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

    memories = memory_manager.recall(
        query,
        k=1,
    )

    print("\nRECALLED:")

    for memory in memories:

        print(
            memory.memory_type,
            "->",
            memory.content,
        )


# ==================================================
# Boundary tests
# ==================================================

boundary_queries = [
    "What is my favorite programming language?",
    "What is Ali's favorite programming language?",
    "What is 25 * 8?",
    "Tell me what Python is.",
]


for query in boundary_queries:

    print("\n==============================")
    print("BOUNDARY QUERY:")
    print(query)

    memories = memory_manager.recall(
        query,
        k=1,
    )

    print("\nRECALLED:")

    if not memories:
        print("[]")
        continue

    for memory in memories:
        print(
            memory.memory_type,
            "->",
            memory.content,
        )        

