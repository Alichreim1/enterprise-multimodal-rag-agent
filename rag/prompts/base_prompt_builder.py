from abc import ABC, abstractmethod

from rag.models.search_result import SearchResult


class BasePromptBuilder(ABC):

    @abstractmethod
    def build(
        self,
        question: str,
        results: list[SearchResult],
    ) -> str:
        pass