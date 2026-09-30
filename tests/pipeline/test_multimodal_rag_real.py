import pytest

from PIL import Image, ImageDraw

from config.bootstrap import build_multimodal_rag_pipeline

from rag.embeddings.image_embedder import ImageEmbedder
from rag.llms.ollama_vision_llm import OllamaVisionLLM


class EmptyTextEmbedder:
    """
    Prevent loading the text embedding model because this
    integration test contains only an image.
    """

    dimension = 3

    def embed_documents(self, chunks):
        return []

    def embed_query(self, query):
        return [0.0, 0.0, 0.0]


@pytest.mark.integration
def test_real_multimodal_rag_pipeline(tmp_path):
    # ==================================================
    # Create Test Image
    # ==================================================

    image_path = tmp_path / "blue_square.png"

    image = Image.new(
        mode="RGB",
        size=(400, 300),
        color="white",
    )

    draw = ImageDraw.Draw(image)

    draw.rectangle(
        (100, 50, 300, 250),
        fill="blue",
    )

    image.save(image_path)

    # ==================================================
    # Real Components
    # ==================================================

    image_embedder = ImageEmbedder(
        model_name=(
            "sentence-transformers/"
            "clip-ViT-B-32"
        ),
    )

    vision_llm = OllamaVisionLLM(
        model="qwen2.5vl:3b",
    )

    # ==================================================
    # Build Pipeline
    # ==================================================

    pipeline = build_multimodal_rag_pipeline(
        file_path=image_path,
        text_embedder=EmptyTextEmbedder(),
        image_embedder=image_embedder,
        vision_llm=vision_llm,
    )

    # ==================================================
    # Ask About Image
    # ==================================================

    response = pipeline.ask(
        "What shape is shown in the image, "
        "and what color is it?"
    )

    print()
    print("=== MULTIMODAL ANSWER ===")
    print(response.answer)

    # ==================================================
    # Assertions
    # ==================================================

    assert isinstance(response.answer, str)
    assert response.answer.strip()

    normalized_answer = response.answer.lower()

    assert "blue" in normalized_answer
    assert (
        "square" in normalized_answer
        or "rectangle" in normalized_answer
    )

    assert response.search_results.has_images
    assert not response.search_results.has_text

    assert len(
        response.search_results.image_results
    ) == 1

    retrieved_image = (
        response.search_results
        .image_results[0]
        .chunk
    )

    assert retrieved_image.is_image
    assert retrieved_image.image_path == str(
        image_path,
    )

    assert response.prompt.has_images
    assert str(image_path) in (
        response.prompt.image_paths
    )

    # ==================================================
    # Performance
    # ==================================================

    print()
    print("=== PERFORMANCE ===")
    print(response.performance)
