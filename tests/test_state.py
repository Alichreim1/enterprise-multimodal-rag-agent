from rag.agent.state import AgentState
from rag.models.task import AgentTask


def test_current_task():

    tasks = [
        AgentTask(
            id=1,
            description="What is Ali's LinkedIn?",
            task_type="retrieve",
        ),
        AgentTask(
            id=2,
            description="What is 246 * 54?",
            task_type="tool",
        ),
    ]

    state = AgentState(
        question=(
            "What is Ali's LinkedIn "
            "and what is 246 * 54?"
        ),
        tasks=tasks,
    )

    assert state.current_task is tasks[0]


def test_move_to_next_task():

    tasks = [
        AgentTask(
            id=1,
            description="What is Ali's LinkedIn?",
            task_type="retrieve",
        ),
        AgentTask(
            id=2,
            description="What is 246 * 54?",
            task_type="tool",
        ),
    ]

    state = AgentState(
        question="test",
        tasks=tasks,
    )

    assert state.current_task.id == 1

    state.move_to_next_task()

    assert state.current_task.id == 2


def test_all_tasks_completed():

    tasks = [
        AgentTask(
            id=1,
            description="Task 1",
            task_type="retrieve",
        ),
        AgentTask(
            id=2,
            description="Task 2",
            task_type="tool",
        ),
    ]

    state = AgentState(
        question="test",
        tasks=tasks,
    )

    assert state.all_tasks_completed is False

    tasks[0].completed = True

    assert state.all_tasks_completed is False

    tasks[1].completed = True

    assert state.all_tasks_completed is True


def test_current_task_none_when_finished():

    tasks = [
        AgentTask(
            id=1,
            description="Task 1",
            task_type="retrieve",
        ),
    ]

    state = AgentState(
        question="test",
        tasks=tasks,
    )

    state.move_to_next_task()

    assert state.current_task is None