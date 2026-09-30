from rag.models.chunk import Chunk
from rag.models.multimodal_search_result import (
    MultimodalSearchResult,
)
from rag.models.search_result import (
    SearchResult,
)
from rag.prompts.multimodal_prompt_builder import (
    MultimodalPromptBuilder,
)


class FakePromptLoader:

    def load(self, template_name: str):

        assert template_name == (
            "multimodal_rag_prompt"
        )

        return """
Text:
{text_context}

Images:
{image_context}

Question:
{question}
"""


def test_multimodal_prompt_builder():

    text_result = SearchResult(
        chunk=Chunk(
            content=(
                "Sales increased in 2026."
            ),
            metadata={
                "source": "report.pdf",
                "page": 1,
            },
        ),
        score=0.9,
    )

    image_result = SearchResult(
        chunk=Chunk(
            content_type="image",
            image_path="images/chart.png",
            mime_type="image/png",
            metadata={
                "source": "chart.png",
                "page": 2,
            },
        ),
        score=0.8,
    )

    results = MultimodalSearchResult(
        text_results=[text_result],
        image_results=[image_result],
    )

    builder = MultimodalPromptBuilder(
        prompt_loader=FakePromptLoader(),
    )

    prompt = builder.build(
        question=(
            "What does the chart show?"
        ),
        results=results,
    )

    assert "Sales increased" in (
        prompt.text
    )

    assert "Image 1" in prompt.text

    assert "chart.png" in prompt.text

    assert (
        "What does the chart show?"
        in prompt.text
    )

    assert prompt.image_paths == [
        "images/chart.png"
    ]

    assert prompt.has_images is True