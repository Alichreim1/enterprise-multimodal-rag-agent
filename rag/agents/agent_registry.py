from rag.agents.base import BaseAgent


class AgentRegistry:

    def __init__(self):
        self._agents: dict[str, BaseAgent] = {}

    def register(
        self,
        task_type: str,
        agent: BaseAgent,
    ) -> None:

        if not task_type:
            raise ValueError(
                "Task type cannot be empty."
            )

        if task_type in self._agents:
            raise ValueError(
                f"Agent already registered "
                f"for task type: {task_type}"
            )

        self._agents[task_type] = agent

    def get(
        self,
        task_type: str,
    ) -> BaseAgent:

        agent = self._agents.get(task_type)

        if agent is None:
            raise ValueError(
                f"No agent registered "
                f"for task type: {task_type}"
            )

        return agent

    def has(
        self,
        task_type: str,
    ) -> bool:

        return task_type in self._agents