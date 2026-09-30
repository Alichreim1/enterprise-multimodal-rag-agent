from rag.chunkers.image_chunker import (
    ImageChunker,
)
from rag.models.document import Document


def test_image_chunker():

    document = Document(
        content_type="image",
        image_path="data/chart.png",
        mime_type="image/png",
        metadata={
            "source": "chart.png",
            "width": 200,
            "height": 100,
        },
    )

    chunker = ImageChunker()

    chunks = chunker.chunk(
        [document]
    )

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk.is_image is True

    assert chunk.image_path == (
        "data/chart.png"
    )

    assert chunk.mime_type == (
        "image/png"
    )

    assert chunk.metadata[
        "content_type"
    ] == "image"

    assert chunk.metadata[
        "chunk_id"
    ] == 0