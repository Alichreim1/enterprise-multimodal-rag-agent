from rag.chunkers.recursive_chunker import RecursiveChunker
from rag.models.document import Document
from rag.models.chunk import Chunk


def test_recursive_chunker_returns_chunks():
    document = Document(
        content=(
            "Artificial intelligence is changing software engineering. "
            "Machine learning allows computers to learn from data. "
            "Retrieval augmented generation improves language model responses. "
            "Embeddings represent text as numerical vectors."
        ),
        metadata={
            "source": "test.txt",
            "page": 1,
            "file_type": "txt",
        },
    )

    chunker = RecursiveChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = chunker.chunk([document])

    assert isinstance(chunks, list)
    assert len(chunks) > 1

    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert chunk.content.strip()


def test_recursive_chunker_preserves_document_metadata():
    document = Document(
        content=(
            "Artificial intelligence is changing software engineering. "
            "Machine learning allows computers to learn from data. "
            "Retrieval augmented generation improves language model responses."
        ),
        metadata={
            "source": "test.pdf",
            "page": 3,
            "file_type": "pdf",
        },
    )

    chunker = RecursiveChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = chunker.chunk([document])

    assert len(chunks) > 0

    for chunk in chunks:
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.metadata["page"] == 3
        assert chunk.metadata["file_type"] == "pdf"


def test_recursive_chunker_assigns_chunk_ids():
    document = Document(
        content=(
            "Artificial intelligence is changing software engineering. "
            "Machine learning allows computers to learn from data. "
            "Retrieval augmented generation improves language model responses. "
            "Embeddings represent text as numerical vectors."
        ),
        metadata={
            "source": "test.txt",
            "page": 1,
            "file_type": "txt",
        },
    )

    chunker = RecursiveChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = chunker.chunk([document])

    chunk_ids = [
        chunk.metadata["chunk_id"]
        for chunk in chunks
    ]

    assert chunk_ids == list(range(len(chunks)))


def test_recursive_chunker_does_not_modify_document_metadata():
    document = Document(
        content=(
            "Artificial intelligence is changing software engineering. "
            "Machine learning allows computers to learn from data. "
            "Retrieval augmented generation improves language model responses."
        ),
        metadata={
            "source": "test.txt",
            "page": 1,
            "file_type": "txt",
        },
    )

    original_metadata = document.metadata.copy()

    chunker = RecursiveChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunker.chunk([document])

    assert document.metadata == original_metadata
    assert "chunk_id" not in document.metadata


def test_recursive_chunker_handles_multiple_documents():
    document1 = Document(
        content=(
            "Document one contains information about artificial intelligence "
            "and machine learning."
        ),
        metadata={
            "source": "document1.pdf",
            "page": 1,
            "file_type": "pdf",
        },
    )

    document2 = Document(
        content=(
            "Document two contains information about retrieval augmented "
            "generation and embeddings."
        ),
        metadata={
            "source": "document2.pdf",
            "page": 2,
            "file_type": "pdf",
        },
    )

    chunker = RecursiveChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = chunker.chunk(
        [document1, document2]
    )

    assert len(chunks) > 0

    document1_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata["source"] == "document1.pdf"
    ]

    document2_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata["source"] == "document2.pdf"
    ]

    assert len(document1_chunks) > 0
    assert len(document2_chunks) > 0


def test_recursive_chunker_handles_short_document():
    document = Document(
        content="Short document.",
        metadata={
            "source": "short.txt",
            "page": 1,
            "file_type": "txt",
        },
    )

    chunker = RecursiveChunker(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = chunker.chunk([document])

    assert len(chunks) == 1

    assert chunks[0].content == "Short document."

    assert chunks[0].metadata["source"] == "short.txt"
    assert chunks[0].metadata["page"] == 1
    assert chunks[0].metadata["file_type"] == "txt"
    assert chunks[0].metadata["chunk_id"] == 0