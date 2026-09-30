from abc import ABC, abstractmethod

from rag.models.document import Document
from rag.models.chunk import Chunk


class BaseChunker(ABC):

    @abstractmethod
    def chunk(
        self,
        documents: list[Document],
    ) -> list[Chunk]:
        """
        Split documents into chunks.
        """
        pass