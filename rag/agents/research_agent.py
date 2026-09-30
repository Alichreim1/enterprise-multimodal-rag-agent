import re

from rag.agents.base import BaseAgent
from rag.models.context import TaskContext
from rag.models.result import AgentResult


class ResearchAgent(BaseAgent):

    def __init__(self, rag , knowledge_base=None,):
        self.rag = rag
        self.knowledge_base = knowledge_base

    @property
    def name(self) -> str:
        return "research_agent"

    def run(self, context: TaskContext) -> AgentResult:

        rag_response = self.rag.ask(
            context.task_description
        )

        answer = rag_response.answer

        email_match = re.search(
            r"[\w\.-]+@[\w\.-]+\.\w+",
            answer,
        )

        data = {
            "answer": answer
        }

        if email_match:
            data["email"] = email_match.group(0)

        return AgentResult(
            status="success",
            data=data,
            message="Research completed successfully.",
        )