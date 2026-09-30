from rag.agent.rag_agent import RAGAgent
from rag.models.planner import PlannerDecision
from rag.models.reflection import ReflectionDecision
from rag.models.task import AgentTask


class FakeTaskDecomposer:

    def decompose(self, question: str):

        return [
            AgentTask(
                id=0,
                description="What is Ali's GitHub?",
                task_type="retrieve",
            ),
            AgentTask(
                id=1,
                description="What is 25 * 8?",
                task_type="tool",
            ),
        ]


class FakePlanner:

    def __init__(self):
        self.calls = []

    def plan(self, task: str):

        self.calls.append(task)

        if task == "What is Ali's GitHub?":
            return PlannerDecision(
                action="retrieve",
                reason="Information must be retrieved.",
            )

        if task == "What is 25 * 8?":
            return PlannerDecision(
                action="tool",
                reason="A calculator is required.",
                tool_name="calculator",
                arguments={
                    "expression": "25 * 8"
                },
            )

        raise ValueError(f"Unexpected task: {task}")


class FakeExecutor:

    def __init__(self):
        self.calls = []

    def execute(self, task, decision):

        self.calls.append(
            (task.description, decision.action)
        )

        if decision.action == "retrieve":
            return type(
                "FakeRAGResponse",
                (),
                {
                    "answer": (
                        "Ali's GitHub is "
                        "https://github.com/Alichreim1/."
                    )
                },
            )()

        if decision.action == "tool":
            return "25 * 8 = 200"

        raise ValueError(
            f"Unexpected action: {decision.action}"
        )


class FakeReflection:

    def __init__(self):
        self.calls = []

    def reflect(self, task: str, result):

        self.calls.append(
            (task, result)
        )

        return ReflectionDecision(
            completed=True,
            reason="The result sufficiently answers the task.",
        )


class FakeFinalAnswerGenerator:

    def __init__(self):
        self.question = None
        self.tasks = None

    def generate(self, question, tasks):

        self.question = question
        self.tasks = tasks

        return (
            "Ali's GitHub is "
            "https://github.com/Alichreim1/. "
            "25 × 8 = 200."
        )

    def generate_stream(self, question, tasks, memories=None):

        self.question = question
        self.tasks = tasks

        answer = (
            "Ali's GitHub is "
            "https://github.com/Alichreim1/. "
            "25 × 8 = 200."
        )

        # Simulate streaming chunks
        yield "Ali's GitHub is "
        yield "https://github.com/Alichreim1/. "
        yield "25 × 8 = 200."


def test_rag_agent_end_to_end():

    decomposer = FakeTaskDecomposer()
    planner = FakePlanner()
    executor = FakeExecutor()
    reflection = FakeReflection()
    final_answer = FakeFinalAnswerGenerator()

    agent = RAGAgent(
        task_decomposer=decomposer,
        planner=planner,
        executor=executor,
        reflection=reflection,
        final_answer_generator=final_answer,
    )

    question = (
        "What is Ali's GitHub and what is 25 * 8?"
    )

    answer = agent.ask(question)

    # ---------------------------------------------------------
    # Final answer
    # ---------------------------------------------------------

    assert answer == (
        "Ali's GitHub is "
        "https://github.com/Alichreim1/. "
        "25 × 8 = 200."
    )

    # ---------------------------------------------------------
    # Planner received both tasks
    # ---------------------------------------------------------

    assert planner.calls == [
        "What is Ali's GitHub?",
        "What is 25 * 8?",
    ]

    # ---------------------------------------------------------
    # Executor executed both actions
    # ---------------------------------------------------------

    assert executor.calls == [
        (
            "What is Ali's GitHub?",
            "retrieve",
        ),
        (
            "What is 25 * 8?",
            "tool",
        ),
    ]

    # ---------------------------------------------------------
    # Reflection evaluated both results
    # ---------------------------------------------------------

    assert len(reflection.calls) == 2

    # ---------------------------------------------------------
    # FinalAnswerGenerator received the original question
    # ---------------------------------------------------------

    assert final_answer.question == question

    # ---------------------------------------------------------
    # Both tasks were completed
    # ---------------------------------------------------------

    assert len(final_answer.tasks) == 2

    assert final_answer.tasks[0].completed is True
    assert final_answer.tasks[1].completed is True

    assert final_answer.tasks[0].result == (
        "Ali's GitHub is "
        "https://github.com/Alichreim1/."
    )

    assert final_answer.tasks[1].result == (
        "25 * 8 = 200"
    )


def test_rag_agent_stream_final_answer():

    decomposer = FakeTaskDecomposer()
    planner = FakePlanner()
    executor = FakeExecutor()
    reflection = FakeReflection()
    final_answer = FakeFinalAnswerGenerator()

    agent = RAGAgent(
        task_decomposer=decomposer,
        planner=planner,
        executor=executor,
        reflection=reflection,
        final_answer_generator=final_answer,
    )

    events = list(
        agent.stream("test question")
    )

    # ---------------------------------------------------------
    # Final answer chunks
    # ---------------------------------------------------------

    chunk_events = [
        event
        for event in events
        if event["type"] == "final_answer_chunk"
    ]

    assert chunk_events

    # ---------------------------------------------------------
    # Reconstruct streamed answer
    # ---------------------------------------------------------

    streamed_text = "".join(
        event["result"]
        for event in chunk_events
    )

    # ---------------------------------------------------------
    # Final answer event
    # ---------------------------------------------------------

    final_event = next(
        event
        for event in events
        if event["type"] == "final_answer"
    )

    assert streamed_text == final_event["result"]
