from rag.models.planner import PlannerDecision
from rag.models.task import AgentTask


class Executor:

    def __init__(
        self,
        rag,
        tool_manager,
    ):
        self.rag = rag
        self.tool_manager = tool_manager

    def execute(
        self,
        task: AgentTask,
        decision: PlannerDecision,
    ):
        if decision.action == "retrieve":
            return self._execute_retrieve(task)

        if decision.action == "tool":
            return self._execute_tool(decision)

        if decision.action == "finish":
            return self._execute_finish(task)

        raise ValueError(
            f"Unsupported planner action: {decision.action}"
        )

    def _execute_retrieve(
        self,
        task: AgentTask,
    ):
        response = self.rag.ask(
            task.description
        )

        return response

    def _execute_tool(
        self,
        decision: PlannerDecision,
    ):
        return self.tool_manager.execute(
            name=decision.tool_name,
            arguments=decision.arguments,
        )

    def _execute_finish(
        self,
        task: AgentTask,
    ):
        if task.result is not None:
            return task.result

        return task.description