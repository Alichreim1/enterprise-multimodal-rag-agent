from rag.embeddings.base_embedder import (
    BaseEmbedder,
)
from rag.models.chunk import Chunk
from rag.models.embedded_chunk import (
    EmbeddedChunk,
)
from rag.retrievers.similarity_retriever import (
    SimilarityRetriever,
)
from rag.vectorstores.faiss_vector_store import (
    FAISSVectorStore,
)


class FakeImageEmbedder(BaseEmbedder):

    def __init__(self):
        self.dimension = 3

    def embed_documents(
        self,
        chunks: list[Chunk],
    ) -> list[EmbeddedChunk]:

        embedded_chunks = []

        for chunk in chunks:

            color = chunk.metadata[
                "color"
            ]

            if color == "red":
                vector = [
                    1.0,
                    0.0,
                    0.0,
                ]

            elif color == "blue":
                vector = [
                    0.0,
                    1.0,
                    0.0,
                ]

            else:
                vector = [
                    0.0,
                    0.0,
                    1.0,
                ]

            embedded_chunks.append(
                EmbeddedChunk(
                    chunk=chunk,
                    vector=vector,
                )
            )

        return embedded_chunks

    def embed_query(
        self,
        query: str,
    ) -> list[float]:

        normalized_query = (
            query.lower()
        )

        if "red" in normalized_query:
            return [
                1.0,
                0.0,
                0.0,
            ]

        if "blue" in normalized_query:
            return [
                0.0,
                1.0,
                0.0,
            ]

        return [
            0.0,
            0.0,
            1.0,
        ]


def test_text_query_retrieves_correct_image():

    red_chunk = Chunk(
        content_type="image",
        image_path="images/red.png",
        mime_type="image/png",
        metadata={
            "source": "red.png",
            "color": "red",
        },
    )

    blue_chunk = Chunk(
        content_type="image",
        image_path="images/blue.png",
        mime_type="image/png",
        metadata={
            "source": "blue.png",
            "color": "blue",
        },
    )

    chunks = [
        red_chunk,
        blue_chunk,
    ]

    embedder = FakeImageEmbedder()

    image_vector_store = (
        FAISSVectorStore(
            dimension=embedder.dimension
        )
    )

    embedded_chunks = (
        embedder.embed_documents(
            chunks
        )
    )

    image_vector_store.add(
        embedded_chunks
    )

    image_retriever = (
        SimilarityRetriever(
            embedder=embedder,
            vector_store=(
                image_vector_store
            ),
            top_k=2,
            min_score=0.0,
        )
    )

    results = image_retriever.retrieve(
        "Find the blue image."
    )

    assert len(results) == 2

    best_result = results[0]

    assert best_result.chunk.is_image
    assert best_result.chunk is blue_chunk

    assert best_result.chunk.image_path == (
        "images/blue.png"
    )

    assert best_result.score == 1.0


def test_image_store_is_separate():

    embedder = FakeImageEmbedder()

    image_vector_store = (
        FAISSVectorStore(
            dimension=embedder.dimension
        )
    )

    assert image_vector_store.size == 0

    chunk = Chunk(
        content_type="image",
        image_path="images/red.png",
        metadata={
            "color": "red",
        },
    )

    image_vector_store.add(
        embedder.embed_documents(
            [chunk]
        )
    )

    assert image_vector_store.size == 1