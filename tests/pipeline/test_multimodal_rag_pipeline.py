from types import SimpleNamespace

import pytest

from rag.models.chunk import Chunk
from rag.models.multimodal_prompt import MultimodalPrompt
from rag.models.multimodal_search_result import MultimodalSearchResult
from rag.models.search_result import SearchResult
from rag.pipeline.multimodal_rag_pipeline import MultimodalRAGPipeline


class FakeMultimodalRetriever:
    def __init__(self, result):
        self.result = result
        self.received_question = None

    def retrieve(self, query: str):
        self.received_question = query
        return self.result


class FakeMultimodalPromptBuilder:
    def __init__(self):
        self.received_question = None
        self.received_results = None

    def build(
        self,
        question,
        results,
    ):
        self.received_question = question
        self.received_results = results

        return MultimodalPrompt(
            text=f"Question: {question}",
            image_paths=[
                "images/chart.png",
            ],
        )


class FakeVisionLLM:
    def __init__(self):
        self.received_prompt = None

    def generate_multimodal(self, prompt):
        self.received_prompt = prompt

        return (
            "The chart shows that "
            "sales increased."
        )

    def generate_multimodal_stream(self, prompt):
        self.received_prompt = prompt

        yield "The chart "
        yield "shows increased sales."


def build_search_results():
    text_result = SearchResult(
        chunk=Chunk(
            content="Sales increased in 2026.",
            content_type="text",
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
                "page": 1,
            },
        ),
        score=0.8,
    )

    return MultimodalSearchResult(
        text_results=[text_result],
        image_results=[image_result],
    )


def test_multimodal_rag_pipeline():
    search_results = build_search_results()

    retriever = FakeMultimodalRetriever(
        search_results,
    )

    prompt_builder = FakeMultimodalPromptBuilder()
    llm = FakeVisionLLM()

    pipeline = MultimodalRAGPipeline(
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
    )

    response = pipeline.ask(
        "What does the chart show?",
    )

    assert (
        retriever.received_question
        == "What does the chart show?"
    )

    assert (
        prompt_builder.received_question
        == "What does the chart show?"
    )

    assert (
        prompt_builder.received_results
        is search_results
    )

    assert (
        llm.received_prompt.image_paths
        == ["images/chart.png"]
    )

    assert response.answer == (
        "The chart shows that "
        "sales increased."
    )

    assert (
        response.search_results
        is search_results
    )

    assert response.prompt.has_images


def test_multimodal_rag_stream():
    search_results = build_search_results()
    llm = FakeVisionLLM()

    pipeline = MultimodalRAGPipeline(
        retriever=FakeMultimodalRetriever(
            search_results,
        ),
        prompt_builder=FakeMultimodalPromptBuilder(),
        llm=llm,
    )

    chunks = list(
        pipeline.ask_stream(
            "What does the chart show?",
        )
    )

    assert chunks == [
        "The chart ",
        "shows increased sales.",
    ]

class FakeRoutingRetriever:

    def __init__(
        self,
        text_results=None,
        image_results=None,
    ):
        self.result = SimpleNamespace(
            text_results=text_results or [],
            image_results=image_results or [],
        )

    def retrieve(
        self,
        question,
    ):
        return self.result


class FakeTextPromptBuilder:

    def build(
        self,
        question,
        results,
    ):
        return (
            f"TEXT PROMPT: {question}"
        )


class FakeRoutingMultimodalPromptBuilder:

    def build(
        self,
        question,
        results,
    ):
        return MultimodalPrompt(
            text=(
                f"MULTIMODAL PROMPT: "
                f"{question}"
            ),
            image_paths=[
                "image.png",
            ],
        )


class FakeTextLLM:

    def generate(
        self,
        prompt,
    ):
        return (
            f"TEXT ANSWER: {prompt}"
        )

    def generate_stream(
        self,
        prompt,
    ):
        yield "TEXT "
        yield "ANSWER"


class FakeRoutingVisionLLM:

    def generate_multimodal(
        self,
        prompt,
    ):
        return (
            f"VISION ANSWER: "
            f"{prompt.text}"
        )

    def generate_multimodal_stream(
        self,
        prompt,
    ):
        yield "VISION "
        yield "ANSWER"


def test_text_prompt_used_without_images():
    pipeline = MultimodalRAGPipeline(
        retriever=FakeRoutingRetriever(
            text_results=[
                object(),
            ],
            image_results=[],
        ),
        prompt_builder=(
            FakeRoutingMultimodalPromptBuilder()
        ),
        llm=FakeRoutingVisionLLM(),
        text_llm=FakeTextLLM(),
        text_prompt_builder=(
            FakeTextPromptBuilder()
        ),
    )

    response = pipeline.ask(
        "What is Ali's LinkedIn?",
    )

    assert response.prompt.has_images is False

    assert response.prompt.text == (
        "TEXT PROMPT: "
        "What is Ali's LinkedIn?"
    )

    assert response.answer == (
        "TEXT ANSWER: "
        "TEXT PROMPT: "
        "What is Ali's LinkedIn?"
    )


def test_text_stream_used_without_images():
    pipeline = MultimodalRAGPipeline(
        retriever=FakeRoutingRetriever(
            text_results=[
                object(),
            ],
            image_results=[],
        ),
        prompt_builder=(
            FakeRoutingMultimodalPromptBuilder()
        ),
        llm=FakeRoutingVisionLLM(),
        text_llm=FakeTextLLM(),
        text_prompt_builder=(
            FakeTextPromptBuilder()
        ),
    )

    chunks = list(
        pipeline.ask_stream(
            "What is Ali's LinkedIn?",
        )
    )

    assert chunks == [
        "TEXT ",
        "ANSWER",
    ]


def test_multimodal_prompt_used_with_images():
    pipeline = MultimodalRAGPipeline(
        retriever=FakeRoutingRetriever(
            text_results=[],
            image_results=[
                object(),
            ],
        ),
        prompt_builder=(
            FakeRoutingMultimodalPromptBuilder()
        ),
        llm=FakeRoutingVisionLLM(),
        text_llm=FakeTextLLM(),
        text_prompt_builder=(
            FakeTextPromptBuilder()
        ),
    )

    response = pipeline.ask(
        "What color is the image?",
    )

    assert response.prompt.has_images is True

    assert response.prompt.text == (
        "MULTIMODAL PROMPT: "
        "What color is the image?"
    )

    assert response.answer == (
        "VISION ANSWER: "
        "MULTIMODAL PROMPT: "
        "What color is the image?"
    )