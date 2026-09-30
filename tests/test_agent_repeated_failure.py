from rag.agent.rag_agent import RAGAgent
from rag.models.task import AgentTask
from rag.models.planner import PlannerDecision
from rag.agent.reflection import ReflectionDecision


class FakeTaskDecomposer:

    def decompose(self, question: str):
        return [
            AgentTask(
                id=1,
                description="What is Ali's favorite movie?",
                task_type="retrieve",
            )
        ]


class FakePlanner:

    def plan(self, task_description: str):
        return PlannerDecision(
            action="retrieve",
            reason="The task requires retrieval.",
        )


class FakeRAG:

    def ask(self, question: str):

        class FakeResponse:
            answer = "Ali's GitHub is https://github.com/Alichreim1/."

        return FakeResponse()


class FakeExecutor:

    def __init__(self, rag, tool_manager=None):
        self.rag = rag

    def execute(self, task, decision):
        return self.rag.ask(task.description)


class AlwaysFailReflection:

    def __init__(self):
        self.calls = 0

    def reflect(self, task_description, result):

        self.calls += 1

        return ReflectionDecision(
            completed=False,
            reason="The result does not answer the requested question.",
        )


class FakeFinalAnswerGenerator:

    def generate(self, question, tasks):
        return "Final answer."


def test_agent_repeated_reflection_failure():

    reflection = AlwaysFailReflection()

    agent = RAGAgent(
        task_decomposer=FakeTaskDecomposer(),
        planner=FakePlanner(),
        executor=FakeExecutor(rag=FakeRAG()),
        reflection=reflection,
        final_answer_generator=FakeFinalAnswerGenerator(),
        max_iterations=3,
    )

    try:
        agent.ask(
            "What is Ali's favorite movie?"
        )

        assert False, "Expected RuntimeError was not raised."

    except RuntimeError as e:

        assert str(e) == "Maximum agent iterations reached."

    # The task should have been attempted exactly 3 times.
    assert reflection.calls == 3