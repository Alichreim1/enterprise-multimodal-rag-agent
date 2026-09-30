from rag.models.context import TaskContext
from rag.agent.state import AgentState
from rag.models.task import AgentTask


def build_task_context(
    state: AgentState,
    task: AgentTask,
) -> TaskContext:

    dependency_results = {
        dependency_id: state.results[dependency_id]
        for dependency_id in task.depends_on
        if dependency_id in state.results
    }

    return TaskContext(
        task_id=task.id,
        task_description=task.description,
        dependency_results=dependency_results,
    )