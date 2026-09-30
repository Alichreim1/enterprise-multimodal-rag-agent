from rag.memory.base_memory import BaseMemory
from rag.memory.models import MemoryItem


class SimpleMemory(BaseMemory):

    def __init__(self):

        self.memories: list[MemoryItem] = []

    def add(
        self,
        memory: MemoryItem,
    ) -> None:

        self.memories.append(memory)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[MemoryItem]:

        query_words = set(
            query.lower().split()
        )

        scored = []

        for memory in self.memories:

            memory_words = set(
                memory.content.lower().split()
            )

            score = len(
                query_words & memory_words
            )

            scored.append(
                (score, memory)
            )

        scored.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        return [
            memory
            for score, memory in scored[:top_k]
            if score > 0
        ]

    def delete(
        self,
        memory_id: str,
    ) -> None:

        self.memories = [
            memory
            for memory in self.memories
            if memory.id != memory_id
        ]

    def clear(self) -> None:

        self.memories.clear()