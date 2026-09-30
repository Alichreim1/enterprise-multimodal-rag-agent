
from rag.agent.state import AgentState
from rag.models.task import AgentTask


def test_agent_state_initialization():

    tasks = [
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

    state = AgentState(
        question="What is Ali's GitHub and what is 25 * 8?",
        tasks=tasks,
    )

    assert state.question == "What is Ali's GitHub and what is 25 * 8?"

    assert len(state.tasks) == 2

    assert state.current_task_index == 0

    assert state.iteration == 0

    assert state.finished is False

    current_task = state.tasks[state.current_task_index]

    assert current_task.id == 0
    assert current_task.description == "What is Ali's GitHub?"
    assert current_task.task_type == "retrieve"

    assert current_task.completed is False
    assert current_task.failed is False
    assert current_task.result is None

