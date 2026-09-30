from rag.models.embedded_chunk import (
    EmbeddedChunk,
)
from rag.pipeline.multimodal_rag_pipeline import (
    MultimodalRAGPipeline,
)

from config.bootstrap import (
    build_knowledge_base_pipeline,
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

    def generate_multimodal(
        self,
        prompt,
    ):
        self.received_prompt = prompt

        return "Knowledge base answer."

    def generate_multimodal_stream(
        self,
        prompt,
    ):
        self.received_prompt = prompt

        yield "Knowledge base "
        yield "answer."


def test_build_shared_knowledge_base_pipeline(
    tmp_path,
):
    # ==================================================
    # Initial File
    # ==================================================

    first_file = tmp_path / "first.txt"

    first_file.write_text(
        "Ali builds RAG systems.",
        encoding="utf-8",
    )

    # ==================================================
    # Build Once
    # ==================================================

    knowledge_base, pipeline = (
        build_knowledge_base_pipeline(
            file_path=first_file,
            text_embedder=FakeTextEmbedder(),
            image_embedder=FakeImageEmbedder(),
            vision_llm=FakeVisionLLM(),
        )
    )

    assert isinstance(
        pipeline,
        MultimodalRAGPipeline,
    )

    assert knowledge_base.document_count == 1
    assert knowledge_base.text_chunk_count == 1
    assert knowledge_base.image_chunk_count == 0

    assert (
        pipeline.retriever
        is knowledge_base.retriever
    )

    # ==================================================
    # Add Another File
    # ==================================================

    second_file = tmp_path / "second.txt"

    second_file.write_text(
        "Sara builds data pipelines.",
        encoding="utf-8",
    )

    knowledge_base.add_file(
        file_path=second_file,
        original_filename="second.txt",
    )

    # ==================================================
    # Same Pipeline Sees Updated Store
    # ==================================================

    assert knowledge_base.document_count == 2
    assert knowledge_base.text_chunk_count == 2

    results = pipeline.retrieve_context(
        "Who builds systems?"
    )

    assert len(results.text_results) == 2

    sources = {
        result.chunk.metadata["source"]
        for result in results.text_results
    }

    assert sources == {
        "first.txt",
        "second.txt",
    }


def test_build_empty_knowledge_base_pipeline():

    knowledge_base, pipeline = (
        build_knowledge_base_pipeline(
            text_embedder=FakeTextEmbedder(),
            image_embedder=FakeImageEmbedder(),
            vision_llm=FakeVisionLLM(),
        )
    )

    assert isinstance(
        pipeline,
        MultimodalRAGPipeline,
    )

    assert knowledge_base.document_count == 0
    assert knowledge_base.text_chunk_count == 0
    assert knowledge_base.image_chunk_count == 0

    results = pipeline.retrieve_context(
        "Any available information?"
    )

    assert not results.has_text
    assert not results.has_images