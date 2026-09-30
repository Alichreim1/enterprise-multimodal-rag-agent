from rag.models.task import AgentTask


def test_agent_task_creation():

    task = AgentTask(
        id=1,
        description="What is Ali's LinkedIn?",
        task_type="retrieve",
    )

    assert task.id == 1
    assert task.description == "What is Ali's LinkedIn?"
    assert task.task_type == "retrieve"

    assert task.completed is False
    assert task.result is None


def test_agent_task_completion():

    task = AgentTask(
        id=1,
        description="What is 246 * 54?",
        task_type="tool",
    )

    task.completed = True
    task.result = "13284"

    assert task.completed is True
    assert task.result == "13284"