import numpy as np
from PIL import Image
import pytest

from rag.embeddings.image_embedder import (
    ImageEmbedder,
)
from rag.models.chunk import Chunk


class FakeImageModel:

    def __init__(self):
        self.received_images = None
        self.received_query = None

    def get_sentence_embedding_dimension(
        self,
    ):
        return 3

    def encode(
        self,
        inputs,
        convert_to_numpy,
        normalize_embeddings,
        show_progress_bar=False,
    ):
        if isinstance(inputs, str):
            self.received_query = inputs

            return np.asarray(
                [0.1, 0.2, 0.3],
                dtype=np.float32,
            )

        self.received_images = inputs

        return np.asarray(
            [
                [0.4, 0.5, 0.6]
                for _ in inputs
            ],
            dtype=np.float32,
        )


def test_image_embedder_embeds_images(
    tmp_path,
):

    image_path = (
        tmp_path / "chart.png"
    )

    Image.new(
        mode="RGB",
        size=(100, 50),
        color="blue",
    ).save(image_path)

    model = FakeImageModel()

    embedder = ImageEmbedder(
        model=model,
    )

    chunk = Chunk(
        content_type="image",
        image_path=str(image_path),
        mime_type="image/png",
        metadata={
            "source": "chart.png",
        },
    )

    embedded_chunks = (
        embedder.embed_documents(
            [chunk]
        )
    )

    assert embedder.dimension == 3
    assert len(embedded_chunks) == 1

    embedded = embedded_chunks[0]

    assert embedded.chunk is chunk

    assert np.allclose(
        embedded.vector,
        [0.4, 0.5, 0.6],
    )

    assert len(
        model.received_images
    ) == 1

    assert model.received_images[
        0
    ].mode == "RGB"


def test_image_embedder_embeds_text_query():

    model = FakeImageModel()

    embedder = ImageEmbedder(
        model=model,
    )

    vector = embedder.embed_query(
        "A blue chart"
    )

    assert model.received_query == (
        "A blue chart"
    )

    assert vector == [
        pytest.approx(0.1),
        pytest.approx(0.2),
        pytest.approx(0.3),
    ]


def test_image_embedder_ignores_text_chunks():

    model = FakeImageModel()

    embedder = ImageEmbedder(
        model=model,
    )

    text_chunk = Chunk(
        content="Text content.",
        content_type="text",
    )

    embedded_chunks = (
        embedder.embed_documents(
            [text_chunk]
        )
    )

    assert embedded_chunks == []
    assert model.received_images is None