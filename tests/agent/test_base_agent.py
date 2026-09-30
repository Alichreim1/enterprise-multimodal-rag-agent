from rag.agents.test_agent import TestAgent
from rag.models.context import TaskContext


def test_test_agent():

    agent = TestAgent()

    context = TaskContext(
        task_id=2,
        task_description="Test task",
    )

    result = agent.run(context)

    assert agent.name == "test_agent"

    assert result.status == "success"

    assert result.data["task_id"] == 2

    assert result.message == "Test agent executed successfully."