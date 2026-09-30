from abc import ABC, abstractmethod

from rag.models.embedded_chunk import EmbeddedChunk
from rag.models.search_result import SearchResult


class BaseVectorStore(ABC):

    @abstractmethod
    def add(
        self,
        embedded_chunks: list[EmbeddedChunk],
    ) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        k: int = 5,
    ) -> list[SearchResult]:
        pass