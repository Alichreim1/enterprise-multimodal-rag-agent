from sentence_transformers import SentenceTransformer
import logging

from rag.embeddings.base_embedder import BaseEmbedder
from rag.models.chunk import Chunk
from rag.models.embedded_chunk import EmbeddedChunk


logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ):
        self.model = SentenceTransformer(model_name)

        self.dimension = (
            self.model.get_embedding_dimension()
        )

    def embed_documents(
        self,
        chunks: list[Chunk],
    ) -> list[EmbeddedChunk]:
        logger.info(
            "Generating embeddings for %d chunks...",
            len(chunks),
        )

        texts = [
            chunk.content
            for chunk in chunks
        ]

        vectors = self.model.encode(
            texts,
            convert_to_numpy=False,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        embedded_chunks = []

        for chunk, vector in zip(chunks, vectors):
            embedded_chunks.append(
                EmbeddedChunk(
                    chunk=chunk,
                    vector=vector,
                )
            )

        logger.info("Embedding generation completed.")

        return embedded_chunks

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        vector = self.model.encode(
            query,
            convert_to_numpy=False,
            normalize_embeddings=True,
        )

        return vector