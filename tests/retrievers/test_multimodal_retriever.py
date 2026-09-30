import pytest

from rag.models.chunk import Chunk
from rag.models.search_result import (
    SearchResult,
)
from rag.retrievers.multimodal_retriever import (
    MultimodalRetriever,
)


class FakeRetriever:

    def __init__(self, results):
        self.results = results
        self.called_with = None
        self.called_top_k = None

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ):
        self.called_with = query
        self.called_top_k = top_k

        return self.results[:top_k]


def test_multimodal_retriever():

    text_chunk = Chunk(
        content=(
            "The chart shows increasing sales."
        ),
        content_type="text",
        metadata={
            "source": "report.pdf",
            "page": 1,
        },
    )

    image_chunk = Chunk(
        content_type="image",
        image_path="images/chart.png",
        mime_type="image/png",
        metadata={
            "source": "chart.png",
            "page": 1,
        },
    )

    text_result = SearchResult(
        chunk=text_chunk,
        score=0.85,
    )

    image_result = SearchResult(
        chunk=image_chunk,
        score=0.72,
    )

    text_retriever = FakeRetriever(
        [text_result]
    )

    image_retriever = FakeRetriever(
        [image_result]
    )

    retriever = MultimodalRetriever(
        text_retriever=text_retriever,
        image_retriever=image_retriever,
        text_top_k=5,
        image_top_k=3,
    )

    result = retriever.retrieve(
        "What does the sales chart show?"
    )

    assert result.has_results is True
    assert result.has_text is True
    assert result.has_images is True

    assert result.text_results == [
        text_result
    ]

    assert result.image_results == [
        image_result
    ]

    assert result.all_results == [
        text_result,
        image_result,
    ]

    assert text_retriever.called_with == (
        "What does the sales chart show?"
    )

    assert image_retriever.called_with == (
        "What does the sales chart show?"
    )

    assert (
        text_retriever.called_top_k
        == 5
    )

    assert (
        image_retriever.called_top_k
        == 3
    )


def test_multimodal_retriever_empty_query():

    retriever = MultimodalRetriever(
        text_retriever=FakeRetriever([]),
        image_retriever=FakeRetriever([]),
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        retriever.retrieve("   ")


def test_multimodal_result_can_be_empty():

    retriever = MultimodalRetriever(
        text_retriever=FakeRetriever([]),
        image_retriever=FakeRetriever([]),
    )

    result = retriever.retrieve(
        "Unknown information"
    )

    assert result.has_results is False
    assert result.has_text is False
    assert result.has_images is False
    assert result.all_results == []