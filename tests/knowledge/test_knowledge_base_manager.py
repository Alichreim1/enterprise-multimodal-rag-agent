from rag.knowledge.knowledge_base_manager import (
    KnowledgeBaseManager,
)
from rag.models.embedded_chunk import EmbeddedChunk
from rag.models.multimodal_prompt import MultimodalPrompt


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


def test_add_two_text_files_to_same_store(
    tmp_path,
):
    # ==================================================
    # Create Files
    # ==================================================

    first_file = tmp_path / "ali.txt"

    first_file.write_text(
        "Ali is an AI engineer.",
        encoding="utf-8",
    )

    second_file = tmp_path / "sara.txt"

    second_file.write_text(
        "Sara is a data scientist.",
        encoding="utf-8",
    )

    # ==================================================
    # Knowledge Base
    # ==================================================

    manager = KnowledgeBaseManager(
        text_embedder=FakeTextEmbedder(),
        image_embedder=FakeImageEmbedder(),
    )

    # ==================================================
    # Add First File
    # ==================================================

    first_result = manager.add_file(
        file_path=first_file,
        original_filename="ali_profile.txt",
    )

    assert manager.document_count == 1
    assert manager.text_chunk_count == 1
    assert manager.image_chunk_count == 0

    # ==================================================
    # Add Second File
    # ==================================================

    second_result = manager.add_file(
        file_path=second_file,
        original_filename="sara_profile.txt",
    )

    assert manager.document_count == 2
    assert manager.text_chunk_count == 2
    assert manager.image_chunk_count == 0

    # ==================================================
    # Different Document IDs
    # ==================================================

    assert (
        first_result["document_id"]
        != second_result["document_id"]
    )

    # ==================================================
    # Both Sources Exist In Same Store
    # ==================================================

    stored_chunks = (
        manager
        .text_vector_store
        .embedded_chunks
    )

    sources = {
        embedded_chunk.chunk.metadata[
            "source"
        ]
        for embedded_chunk in stored_chunks
    }

    assert sources == {
        "ali_profile.txt",
        "sara_profile.txt",
    }

    # ==================================================
    # Stored Metadata
    # ==================================================

    for embedded_chunk in stored_chunks:

        metadata = (
            embedded_chunk.chunk.metadata
        )

        assert "document_id" in metadata
        assert "stored_path" in metadata
        assert "original_filename" in metadata


def test_document_records_are_available(
    tmp_path,
):
    file_path = tmp_path / "document.txt"

    file_path.write_text(
        "Enterprise RAG system.",
        encoding="utf-8",
    )

    manager = KnowledgeBaseManager(
        text_embedder=FakeTextEmbedder(),
        image_embedder=FakeImageEmbedder(),
    )

    result = manager.add_file(
        file_path=file_path,
        original_filename="enterprise.txt",
    )

    assert manager.documents == [
        result
    ]

    assert result["filename"] == (
        "enterprise.txt"
    )

    assert result["text_chunks"] == 1
    assert result["image_chunks"] == 0



class FakeMultimodalPromptBuilder:

    def build(
        self,
        question,
        results,
    ):
        text_sources = [
            result.chunk.metadata["source"]
            for result in results.text_results
        ]

        image_sources = [
            result.chunk.metadata["source"]
            for result in results.image_results
        ]

        return MultimodalPrompt(
            text=(
                f"Question: {question}\n"
                f"Text sources: {text_sources}\n"
                f"Image sources: {image_sources}"
            ),
            image_paths=[
                result.chunk.image_path
                for result in results.image_results
            ],
        )


class FakeVisionLLM:

    def __init__(self):
        self.received_prompt = None

    def generate_multimodal(
        self,
        prompt,
    ):
        self.received_prompt = prompt

        return prompt.text

    def generate_multimodal_stream(
        self,
        prompt,
    ):
        self.received_prompt = prompt

        yield prompt.text