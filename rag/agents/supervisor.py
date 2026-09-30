from rag.agent.context_builder import build_task_context
from rag.agent.state import AgentState
from rag.agents.agent_registry import AgentRegistry
from rag.models.result import AgentResult
from rag.models.task import AgentTask


class Supervisor:

    def __init__(
        self,
        registry: AgentRegistry,
    ):
        self.registry = registry

    def get_current_task(
        self,
        state: AgentState,
    ) -> AgentTask | None:

        if state.current_task_index >= len(state.tasks):
            return None

        return state.tasks[state.current_task_index]

    def dependencies_satisfied(
        self,
        state: AgentState,
        task: AgentTask,
    ) -> bool:

        for dependency_id in task.depends_on:
            result = state.results.get(dependency_id)

            if result is None:
                return False

            if result.status != "success":
                return False

        return True

    def can_handle(
        self,
        task_type: str,
    ) -> bool:

        return self.registry.has(task_type)

    def execute_current_task(
        self,
        state: AgentState,
    ) -> AgentResult:

        task = self.get_current_task(state)

        if task is None:
            raise RuntimeError(
                "No current task available."
            )

        if not self.dependencies_satisfied(
            state,
            task,
        ):
            raise RuntimeError(
                f"Dependencies are not satisfied "
                f"for task: {task.id}"
            )

        agent = self.registry.get(
            task.task_type
        )

        context = build_task_context(
            state=state,
            task=task,
        )

        task.attempts += 1
        state.iteration += 1

        result = agent.run(context)

        state.results[task.id] = result
        task.result = result

        if result.status == "success":
            task.completed = True
            task.failed = False
        else:
            task.completed = False
            task.failed = True

        state.current_task_index += 1

        return result

    def run(
        self,
        state: AgentState,
    ) -> AgentState:

        while self.get_current_task(state) is not None:
            result = self.execute_current_task(state)

            if result.status != "success":
                state.finished = True
                return state

        state.finished = True

        return state