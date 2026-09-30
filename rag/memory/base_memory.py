from abc import ABC, abstractmethod

from rag.memory.models import MemoryItem


class BaseMemory(ABC):

    @abstractmethod
    def add(
        self,
        memory: MemoryItem,
    ) -> None:
        pass

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[MemoryItem]:
        pass

    @abstractmethod
    def delete(
        self,
        memory_id: str,
    ) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass