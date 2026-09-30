from PIL import Image

from config.bootstrap import (
    build_multimodal_rag_pipeline,
)
from rag.models.embedded_chunk import EmbeddedChunk
from rag.pipeline.multimodal_rag_pipeline import (
    MultimodalRAGPipeline,
)


class FakeTextEmbedder:

    dimension = 3

    def embed_documents(self, chunks):
        return [
            EmbeddedChunk(
                chunk=chunk,
                vector=[1.0, 0.0, 0.0],
            )
            for chunk in chunks
        ]

    def embed_query(self, query):
        return [1.0, 0.0, 0.0]


class FakeImageEmbedder:

    dimension = 3

    def embed_documents(self, chunks):
        return [
            EmbeddedChunk(
                chunk=chunk,
                vector=[0.0, 1.0, 0.0],
            )
            for chunk in chunks
        ]

    def embed_query(self, query):
        return [0.0, 1.0, 0.0]


class FakeVisionLLM:

    def __init__(self):
        self.received_prompt = None

    def generate_multimodal(self, prompt):
        self.received_prompt = prompt

        return "The image contains a blue square."

    def generate_multimodal_stream(self, prompt):
        self.received_prompt = prompt

        yield "The image "
        yield "contains a blue square."


def test_build_multimodal_rag_pipeline(tmp_path):

    image_path = tmp_path / "blue_square.png"

    Image.new(
        mode="RGB",
        size=(100, 100),
        color="blue",
    ).save(image_path)

    vision_llm = FakeVisionLLM()

    pipeline = build_multimodal_rag_pipeline(
        file_path=image_path,
        text_embedder=FakeTextEmbedder(),
        image_embedder=FakeImageEmbedder(),
        vision_llm=vision_llm,
    )

    assert isinstance(
        pipeline,
        MultimodalRAGPipeline,
    )

    response = pipeline.ask(
        "What is shown in the image?"
    )

    assert response.answer == (
        "The image contains a blue square."
    )

    assert response.search_results.has_images
    assert not response.search_results.has_text

    assert len(
        response.search_results.image_results
    ) == 1

    image_result = (
        response.search_results.image_results[0]
    )

    assert image_result.chunk.is_image
    assert image_result.chunk.image_path == str(
        image_path
    )

    assert response.prompt.has_images
    assert response.prompt.image_paths == [
        str(image_path)
    ]

    assert vision_llm.received_prompt is not None