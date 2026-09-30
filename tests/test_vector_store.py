

from rag.models.chunk import Chunk
from rag.models.embedded_chunk import EmbeddedChunk
from rag.vectorstores.faiss_vector_store import FAISSVectorStore


def create_embedded_chunk(
    content: str,
    vector: list[float],
    chunk_id: int,
) -> EmbeddedChunk:

    chunk = Chunk(
        content=content,
        metadata={
            "source": "test.txt",
            "page": 1,
            "file_type": "txt",
            "chunk_id": chunk_id,
        },
    )

    return EmbeddedChunk(
        chunk=chunk,
        vector=vector,
    )


def test_vector_store_starts_empty():
    store = FAISSVectorStore(dimension=3)

    assert store.size == 0


def test_vector_store_adds_embeddings():
    store = FAISSVectorStore(dimension=3)

    embedded_chunks = [
        create_embedded_chunk(
            content="First chunk",
            vector=[1.0, 0.0, 0.0],
            chunk_id=0,
        ),
        create_embedded_chunk(
            content="Second chunk",
            vector=[0.0, 1.0, 0.0],
            chunk_id=1,
        ),
    ]

    store.add(embedded_chunks)

    assert store.size == 2


def test_vector_store_search_returns_most_similar_chunk():
    store = FAISSVectorStore(dimension=3)

    embedded_chunks = [
        create_embedded_chunk(
            content="Artificial intelligence",
            vector=[1.0, 0.0, 0.0],
            chunk_id=0,
        ),
        create_embedded_chunk(
            content="Machine learning",
            vector=[0.0, 1.0, 0.0],
            chunk_id=1,
        ),
        create_embedded_chunk(
            content="Cooking recipes",
            vector=[0.0, 0.0, 1.0],
            chunk_id=2,
        ),
    ]

    store.add(embedded_chunks)

    query_vector = [1.0, 0.0, 0.0]

    results = store.search(
        query_vector=query_vector,
        k=1,
    )

    assert len(results) == 1

    assert (
        results[0].chunk.content
        == "Artificial intelligence"
    )

    assert results[0].score == 1.0


def test_vector_store_returns_results_in_similarity_order():
    store = FAISSVectorStore(dimension=3)

    embedded_chunks = [
        create_embedded_chunk(
            content="Low similarity",
            vector=[0.5, 0.5, 0.0],
            chunk_id=0,
        ),
        create_embedded_chunk(
            content="Highest similarity",
            vector=[1.0, 0.0, 0.0],
            chunk_id=1,
        ),
        create_embedded_chunk(
            content="No similarity",
            vector=[0.0, 0.0, 1.0],
            chunk_id=2,
        ),
    ]

    store.add(embedded_chunks)

    query_vector = [1.0, 0.0, 0.0]

    results = store.search(
        query_vector=query_vector,
        k=3,
    )

    assert len(results) == 3

    assert (
        results[0].chunk.content
        == "Highest similarity"
    )

    assert results[0].score > results[1].score
    assert results[1].score > results[2].score


def test_vector_store_returns_empty_when_no_vectors_exist():
    store = FAISSVectorStore(dimension=3)

    results = store.search(
        query_vector=[1.0, 0.0, 0.0],
        k=3,
    )

    assert results == []


def test_vector_store_clear():
    store = FAISSVectorStore(dimension=3)

    embedded_chunks = [
        create_embedded_chunk(
            content="First chunk",
            vector=[1.0, 0.0, 0.0],
            chunk_id=0,
        ),
    ]

    store.add(embedded_chunks)

    assert store.size == 1

    store.clear()

    assert store.size == 0

    results = store.search(
        query_vector=[1.0, 0.0, 0.0],
        k=1,
    )

    assert results == []