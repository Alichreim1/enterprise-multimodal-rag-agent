from abc import ABC, abstractmethod

from rag.models.chunk import Chunk
from rag.models.embedded_chunk import EmbeddedChunk


class BaseEmbedder(ABC):

    @abstractmethod
    def embed_documents(
        self,
        chunks: list[Chunk],
    ) -> list[EmbeddedChunk]:
        """
        Generate embeddings for a list of chunks.
        """
        pass

    @abstractmethod
    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """
        Generate an embedding for a query.
        """
        pass