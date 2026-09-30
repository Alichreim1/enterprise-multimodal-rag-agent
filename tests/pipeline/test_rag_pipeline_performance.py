import pytest

from rag.observability.performance_tracker import (
    PerformanceTracker,
)
from rag.pipeline.rag_pipeline import (
    RAGPipeline,
)


class FakeClock:

    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


class FakeRetriever:

    def retrieve(self, question):

        return [
            "retrieved context"
        ]


class FakePromptBuilder:

    def build(
        self,
        question,
        results,
    ):
        return (
            f"Question: {question}\n"
            f"Context: {results}"
        )


class FakeLLM:

    def generate(self, prompt):

        return "Final RAG answer."


def test_rag_pipeline_records_performance():

    clock = FakeClock(
        [
            0.0,  # tracker starts

            0.1,  # retrieval starts
            0.3,  # retrieval stops

            0.4,  # prompt building starts
            0.5,  # prompt building stops

            0.6,  # LLM generation starts
            1.0,  # LLM generation stops

            1.2,  # tracker stops
        ]
    )

    def tracker_factory():

        return PerformanceTracker(
            clock=clock
        )

    pipeline = RAGPipeline(
        retriever=FakeRetriever(),
        prompt_builder=FakePromptBuilder(),
        llm=FakeLLM(),
        performance_tracker_factory=(
            tracker_factory
        ),
    )

    response = pipeline.ask(
        "What is RAG?"
    )

    assert response.answer == (
        "Final RAG answer."
    )

    performance = response.performance

    assert performance["finished"] is True

    assert performance["total_ms"] == (
        pytest.approx(1200.0)
    )

    assert (
        performance["stages"][
            "retrieval"
        ]["total_ms"]
        == pytest.approx(200.0)
    )

    assert (
        performance["stages"][
            "prompt_building"
        ]["total_ms"]
        == pytest.approx(100.0)
    )

    assert (
        performance["stages"][
            "llm_generation"
        ]["total_ms"]
        == pytest.approx(400.0)
    )


def test_rag_response_performance_is_independent():

    first_pipeline = RAGPipeline(
        retriever=FakeRetriever(),
        prompt_builder=FakePromptBuilder(),
        llm=FakeLLM(),
    )

    second_pipeline = RAGPipeline(
        retriever=FakeRetriever(),
        prompt_builder=FakePromptBuilder(),
        llm=FakeLLM(),
    )

    first_response = first_pipeline.ask(
        "First question"
    )

    second_response = second_pipeline.ask(
        "Second question"
    )

    assert first_response.performance
    assert second_response.performance

    assert (
        first_response.performance
        is not second_response.performance
    )