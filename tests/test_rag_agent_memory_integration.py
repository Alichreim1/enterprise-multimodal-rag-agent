from rag.agent.rag_agent import RAGAgent
from rag.models.task import AgentTask
from rag.memory.models import MemoryItem


class FakeMemoryManager:

    def __init__(self, memories):
        self.memories = memories
        self.calls = []

    def recall_relevant(self, query: str, k: int = 5):
        self.calls.append((query, k))
        return self.memories


class FakeTaskDecomposer:

    def decompose(self, question: str):
        return []


class FakePlanner:
    pass


class FakeExecutor:
    pass


class FakeReflection:
    pass


class FakeFinalAnswerGenerator:

    def __init__(self):
        self.question = None
        self.tasks = None
        self.memories = None

    def generate(
        self,
        question,
        tasks,
        memories,
    ):
        self.question = question
        self.tasks = tasks
        self.memories = memories

        return "Python"


def test_agent_passes_relevant_memories_to_final_answer():

    memory = MemoryItem(
        id=1,
        content="User's favorite programming language is Python.",
        memory_type="preference",
    )

    memory_manager = FakeMemoryManager(
        memories=[memory]
    )

    final_answer_generator = FakeFinalAnswerGenerator()

    agent = RAGAgent(
        task_decomposer=FakeTaskDecomposer(),
        planner=FakePlanner(),
        executor=FakeExecutor(),
        reflection=FakeReflection(),
        final_answer_generator=final_answer_generator,
        memory_manager=memory_manager,
    )

    answer = agent.ask(
        "What is my favorite programming language?"
    )

    assert answer == "Python"

    assert memory_manager.calls == [
        ("What is my favorite programming language?", 5)
    ]

    assert final_answer_generator.memories == [
        memory
    ]