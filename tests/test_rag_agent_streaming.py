from rag.agent.rag_agent import RAGAgent


class FakeMemoryManager:

    def recall_relevant(self, query: str, k: int):
        return []

    def remember(self, content: str):
        pass


class FakeTaskDecomposer:

    def decompose(self, question: str):
        return []


class FakeFinalAnswerGenerator:

    def generate(
        self,
        question: str,
        tasks,
        memories,
    ):
        return "Complete final answer."

    def generate_stream(
        self,
        question: str,
        tasks,
        memories,
    ):
        yield "Complete"
        yield " final"
        yield " answer."


class FakePlanner:
    pass


class FakeExecutor:
    pass


class FakeReflection:
    pass


def test_rag_agent_streams_final_answer():

    agent = RAGAgent(
        task_decomposer=FakeTaskDecomposer(),
        planner=FakePlanner(),
        executor=FakeExecutor(),
        reflection=FakeReflection(),
        final_answer_generator=FakeFinalAnswerGenerator(),
        memory_manager=FakeMemoryManager(),
    )

    events = list(
        agent.stream(
            "What is a test question?"
        )
    )

    # -----------------------------------------------------
    # Final Answer Chunks
    # -----------------------------------------------------

    chunk_events = [
        event
        for event in events
        if event["type"] == "final_answer_chunk"
    ]

    assert chunk_events

    streamed_text = "".join(
        event["result"]
        for event in chunk_events
    )

    assert streamed_text == "Complete final answer."

    # -----------------------------------------------------
    # Final Answer Event
    # -----------------------------------------------------

    final_answer_event = next(
        event
        for event in events
        if event["type"] == "final_answer"
    )

    assert final_answer_event["result"] == (
        "Complete final answer."
    )

    # -----------------------------------------------------
    # Event Order
    # -----------------------------------------------------

    event_types = [
        event["type"]
        for event in events
    ]

    assert (
        event_types.index("final_answer_started")
        <
        event_types.index("final_answer_chunk")
    )

    assert (
        event_types.index("final_answer_chunk")
        <
        event_types.index("final_answer")
    )

    assert (
        event_types.index("final_answer")
        <
        event_types.index("agent_finished")
    )