from rag.chunkers.recursive_chunker import RecursiveChunker
from rag.models.document import Document


def test_chunk_creation():

    document = Document(

        content="Hello World " * 500,

        metadata={
            "page": 1,
            "source": "test.pdf",
            "file_type": "pdf",
        },

    )

    chunker = RecursiveChunker(

        chunk_size=100,

        chunk_overlap=20,

    )

    chunks = chunker.chunk(
        [document]
    )

    assert len(chunks) > 1


def test_chunk_metadata():

    document = Document(

        content="Hello " * 200,

        metadata={
            "page": 1,
            "source": "test.pdf",
            "file_type": "pdf",
        },

    )

    chunker = RecursiveChunker()

    chunks = chunker.chunk(
        [document]
    )

    assert chunks[0].metadata["page"] == 1

    assert chunks[0].metadata["source"] == "test.pdf"

    assert "chunk_id" in chunks[0].metadata