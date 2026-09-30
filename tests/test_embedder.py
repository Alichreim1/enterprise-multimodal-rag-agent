import math

from rag.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from rag.models.chunk import Chunk
from rag.models.embedded_chunk import EmbeddedChunk


MODEL_NAME = "BAAI/bge-small-en-v1.5"


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    norm_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    norm_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    return dot_product / (norm_a * norm_b)


def test_embedder_initializes():
    embedder = SentenceTransformerEmbedder(
        MODEL_NAME
    )

    assert embedder.model is not None
    assert embedder.dimension > 0


def test_embed_documents_returns_embedded_chunks():
    embedder = SentenceTransformerEmbedder(
        MODEL_NAME
    )

    chunks = [
        Chunk(
            content="Artificial intelligence is changing software engineering.",
            metadata={
                "source": "test.txt",
                "page": 1,
                "chunk_id": 0,
            },
        ),
        Chunk(
            content="Machine learning allows computers to learn from data.",
            metadata={
                "source": "test.txt",
                "page": 1,
                "chunk_id": 1,
            },
        ),
    ]

    embedded_chunks = embedder.embed_documents(
        chunks
    )

    assert isinstance(embedded_chunks, list)
    assert len(embedded_chunks) == len(chunks)

    for embedded_chunk in embedded_chunks:
        assert isinstance(
            embedded_chunk,
            EmbeddedChunk,
        )

        assert embedded_chunk.chunk is not None
        assert embedded_chunk.vector is not None


def test_embedding_dimension_is_consistent():
    embedder = SentenceTransformerEmbedder(
        MODEL_NAME
    )

    chunks = [
        Chunk(
            content="Artificial intelligence.",
        ),
        Chunk(
            content="Machine learning.",
        ),
        Chunk(
            content="Retrieval augmented generation.",
        ),
    ]

    embedded_chunks = embedder.embed_documents(
        chunks
    )

    for embedded_chunk in embedded_chunks:
        assert len(
            embedded_chunk.vector
        ) == embedder.dimension


def test_embeddings_are_normalized():
    embedder = SentenceTransformerEmbedder(
        MODEL_NAME
    )

    chunks = [
        Chunk(
            content="Artificial intelligence is changing software engineering."
        ),
        Chunk(
            content="Machine learning allows computers to learn from data."
        ),
    ]

    embedded_chunks = embedder.embed_documents(
        chunks
    )

    for embedded_chunk in embedded_chunks:

        norm = math.sqrt(
            sum(
                value * value
                for value in embedded_chunk.vector
            )
        )

        assert math.isclose(
            norm,
            1.0,
            rel_tol=1e-5,
            abs_tol=1e-5,
        )


def test_embed_query_returns_correct_dimension():
    embedder = SentenceTransformerEmbedder(
        MODEL_NAME
    )

    query = "How does artificial intelligence work?"

    vector = embedder.embed_query(query)

    assert vector is not None
    assert len(vector) == embedder.dimension


def test_similar_texts_have_higher_similarity():
    embedder = SentenceTransformerEmbedder(
        MODEL_NAME
    )

    query = (
        "Employees are allowed to take annual leave."
    )

    similar_text = (
        "Employees can take vacation days from work."
    )

    unrelated_text = (
        "The company uses Python for backend development."
    )

    query_vector = embedder.embed_query(query)

    similar_vector = embedder.embed_query(
        similar_text
    )

    unrelated_vector = embedder.embed_query(
        unrelated_text
    )

    similar_similarity = cosine_similarity(
        query_vector,
        similar_vector,
    )

    unrelated_similarity = cosine_similarity(
        query_vector,
        unrelated_vector,
    )

    assert similar_similarity > unrelated_similarity