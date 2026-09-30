from rag.agents.base import BaseAgent
from rag.models.context import TaskContext
from rag.models.result import AgentResult


class TestAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "test_agent"

    def run(self, context: TaskContext) -> AgentResult:
        return AgentResult(
            status="success",
            data={
                "task_id": context.task_id
            },
            message="Test agent executed successfully.",
        )