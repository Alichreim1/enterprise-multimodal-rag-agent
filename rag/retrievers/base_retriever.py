from abc import ABC, abstractmethod

from rag.models.search_result import SearchResult


class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(
        self,
        query: str,
    ) -> list[SearchResult]:
        """
        Retrieve the most relevant chunks.
        """
        pass