from rag.agents.base import BaseAgent
from rag.models.context import TaskContext
from rag.models.result import AgentResult


class WebResearchAgent(BaseAgent):

    def __init__(
        self,
        web_search_tool,
    ):
        self.web_search_tool = (
            web_search_tool
        )

    @property
    def name(self) -> str:
        return "web_research_agent"

    def run(
        self,
        context: TaskContext,
    ) -> AgentResult:

        query = (
            context.task_description
            .strip()
        )

        if not query:
            return AgentResult(
                status="failure",
                message=(
                    "Web research query "
                    "cannot be empty."
                ),
            )

        try:
            answer = (
                self.web_search_tool.run(
                    query=query,
                )
            )

        except Exception as exc:
            return AgentResult(
                status="failure",
                data={
                    "query": query,
                },
                message=(
                    "Web research failed: "
                    f"{exc}"
                ),
            )

        if not answer or not answer.strip():
            return AgentResult(
                status="failure",
                data={
                    "query": query,
                },
                message=(
                    "Web search returned "
                    "an empty result."
                ),
            )

        return AgentResult(
            status="success",
            data={
                "query": query,
                "answer": answer.strip(),
            },
            message=(
                "Web research completed "
                "successfully."
            ),
        )