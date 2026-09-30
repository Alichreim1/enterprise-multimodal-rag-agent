from pathlib import Path

from rag.embeddings.base_embedder import BaseEmbedder
from rag.memory.models import MemoryItem
from rag.models.chunk import Chunk
from rag.vectorstores.base_vector_store import BaseVectorStore


class VectorMemory:

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
    ):
        self.embedder = embedder
        self.vector_store = vector_store

    # ==================================================
    # Add memory
    # ==================================================

    def add(
        self,
        memory: MemoryItem,
    ) -> None:
        chunk = Chunk(
            content=memory.content,
            metadata={
                "memory_id": memory.id,
                "memory_type": memory.memory_type,
                "created_at": (
                    memory.created_at.isoformat()
                    if memory.created_at
                    else None
                ),
            },
        )

        embedded_chunks = self.embedder.embed_documents(
            [chunk]
        )

        self.vector_store.add(
            embedded_chunks
        )

    # ==================================================
    # Add many memories
    # ==================================================

    def add_many(
        self,
        memories: list[MemoryItem],
    ) -> None:
        for memory in memories:
            self.add(memory)

    # ==================================================
    # Retrieve memories
    # ==================================================

    def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> list[MemoryItem]:
        if self.vector_store.size == 0:
            return []

        query_vector = self.embedder.embed_query(
            query
        )

        results = self.vector_store.search(
            query_vector,
            k=k,
        )

        memories = []

        for result in results:
            metadata = result.chunk.metadata

            memories.append(
                MemoryItem(
                    id=metadata.get("memory_id"),
                    content=result.chunk.content,
                    memory_type=metadata.get(
                        "memory_type",
                        "unknown",
                    ),
                    created_at=None,
                )
            )

        return memories

    # ==================================================
    # Retrieve memories with scores
    # ==================================================

    def retrieve_with_scores(
        self,
        query: str,
        k: int = 5,
    ):
        if self.vector_store.size == 0:
            return []

        query_vector = self.embedder.embed_query(
            query
        )

        return self.vector_store.search(
            query_vector,
            k=k,
        )

    # ==================================================
    # Size
    # ==================================================

    @property
    def size(self) -> int:
        return self.vector_store.size

    # ==================================================
    # Save
    # ==================================================

    def save(
        self,
        index_path: Path,
        metadata_path: Path,
    ) -> None:
        index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.vector_store.save(
            index_path=index_path,
            metadata_path=metadata_path,
        )

    # ==================================================
    # Load
    # ==================================================

    def load(
        self,
        index_path: Path,
        metadata_path: Path,
    ) -> None:
        self.vector_store.load(
            index_path=index_path,
            metadata_path=metadata_path,
        )

    # ==================================================
    # Exists
    # ==================================================

    def exists(
        self,
        index_path: Path,
        metadata_path: Path,
    ) -> bool:
        return self.vector_store.exists(
            index_path=index_path,
            metadata_path=metadata_path,
        )

    # ==================================================
    # Clear
    # ==================================================

    def clear(self) -> None:
        self.vector_store.clear()
