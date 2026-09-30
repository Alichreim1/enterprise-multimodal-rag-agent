import pytest

from rag.agent.rag_agent import RAGAgent
from rag.observability.performance_tracker import (
    PerformanceTracker,
)


class FakeClock:

    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


class FakeMemoryManager:

    def recall_relevant(
        self,
        query,
        k,
    ):
        return []


class FakeTaskDecomposer:

    def decompose(self, question):
        return []


class FakeFinalAnswerGenerator:

    def generate(
        self,
        question,
        tasks,
        memories,
    ):
        return "Final answer."


def test_rag_agent_records_complete_performance():

    clock = FakeClock(
        [
            0.0,  # total starts

            0.1,  # memory starts
            0.4,  # memory stops

            0.5,  # decomposition starts
            0.9,  # decomposition stops

            1.0,  # final answer starts
            1.6,  # final answer stops

            2.0,  # total stops
        ]
    )

    def tracker_factory():

        return PerformanceTracker(
            clock=clock
        )

    agent = RAGAgent(
        task_decomposer=FakeTaskDecomposer(),
        planner=object(),
        executor=object(),
        reflection=object(),
        final_answer_generator=(
            FakeFinalAnswerGenerator()
        ),
        memory_manager=FakeMemoryManager(),
        performance_tracker_factory=(
            tracker_factory
        ),
    )

    answer = agent.ask(
        "Test question"
    )

    assert answer == "Final answer."

    performance = agent.last_performance

    assert performance["total_ms"] == (
        pytest.approx(2000.0)
    )

    assert (
        performance["stages"][
            "memory_recall"
        ]["total_ms"]
        == pytest.approx(300.0)
    )

    assert (
        performance["stages"][
            "task_decomposition"
        ]["total_ms"]
        == pytest.approx(400.0)
    )

    assert (
        performance["stages"][
            "final_answer_generation"
        ]["total_ms"]
        == pytest.approx(600.0)
    )