from rag.agent.task_decomposer import TaskDecomposer
from rag.models.task import AgentTask


class FakeLLM:

    def __init__(self, response):
        self.response = response

    def decide(self, prompt):
        return self.response


def test_decompose_multiple_tasks():

    llm = FakeLLM(
        """
        [
            {
                "id": 1,
                "description": "What is Ali's LinkedIn?",
                "task_type": "retrieve"
            },
            {
                "id": 2,
                "description": "What is 246 * 54?",
                "task_type": "tool"
            }
        ]
        """
    )

    decomposer = TaskDecomposer(llm)

    tasks = decomposer.decompose(
        "What is Ali's LinkedIn and what is 246 * 54?"
    )

    assert len(tasks) == 2

    assert isinstance(
        tasks[0],
        AgentTask,
    )

    assert tasks[0].id == 1
    assert (
        tasks[0].description
        == "What is Ali's LinkedIn?"
    )
    assert tasks[0].task_type == "retrieve"

    assert tasks[1].id == 2
    assert (
        tasks[1].description
        == "What is 246 * 54?"
    )
    assert tasks[1].task_type == "tool"


def test_decompose_single_retrieval_task():

    llm = FakeLLM(
        """
        [
            {
                "id": 1,
                "description": "What is Ali's experience?",
                "task_type": "retrieve"
            }
        ]
        """
    )

    decomposer = TaskDecomposer(llm)

    tasks = decomposer.decompose(
        "What is Ali's experience?"
    )

    assert len(tasks) == 1

    assert tasks[0].task_type == "retrieve"


def test_decompose_single_tool_task():

    llm = FakeLLM(
        """
        [
            {
                "id": 1,
                "description": "What is 25 * 4?",
                "task_type": "tool"
            }
        ]
        """
    )

    decomposer = TaskDecomposer(llm)

    tasks = decomposer.decompose(
        "What is 25 * 4?"
    )

    assert len(tasks) == 1

    assert tasks[0].task_type == "tool"

    assert (
        tasks[0].description
        == "What is 25 * 4?"
    )


def test_tasks_start_incomplete():

    llm = FakeLLM(
        """
        [
            {
                "id": 1,
                "description": "What is Ali's LinkedIn?",
                "task_type": "retrieve"
            }
        ]
        """
    )

    decomposer = TaskDecomposer(llm)

    tasks = decomposer.decompose(
        "What is Ali's LinkedIn?"
    )

    assert tasks[0].completed is False
    assert tasks[0].result is None