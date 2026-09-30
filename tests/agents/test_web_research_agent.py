from rag.agents.web_research_agent import (
    WebResearchAgent,
)
from rag.models.context import TaskContext


class FakeWebSearchTool:

    def __init__(
        self,
        result: str,
    ):
        self.result = result
        self.called_with = None

    def run(
        self,
        query: str,
    ):
        self.called_with = query

        return self.result


def test_web_research_agent():

    tool = FakeWebSearchTool(
        result=(
            "The latest stable Python "
            "version is Python 3.14."
        ),
    )

    agent = WebResearchAgent(
        web_search_tool=tool,
    )

    context = TaskContext(
        task_id=1,
        task_description=(
            "What is the latest stable "
            "version of Python?"
        ),
    )

    result = agent.run(context)

    assert result.status == "success"

    assert tool.called_with == (
        "What is the latest stable "
        "version of Python?"
    )

    assert result.data == {
        "query": (
            "What is the latest stable "
            "version of Python?"
        ),
        "answer": (
            "The latest stable Python "
            "version is Python 3.14."
        ),
    }

    assert result.message == (
        "Web research completed "
        "successfully."
    )


def test_web_research_agent_rejects_empty_query():

    tool = FakeWebSearchTool(
        result="Unused result.",
    )

    agent = WebResearchAgent(
        web_search_tool=tool,
    )

    context = TaskContext(
        task_id=1,
        task_description="   ",
    )

    result = agent.run(context)

    assert result.status == "failure"

    assert result.message == (
        "Web research query cannot be empty."
    )

    assert tool.called_with is None


def test_web_research_agent_handles_empty_result():

    tool = FakeWebSearchTool(
        result="   ",
    )

    agent = WebResearchAgent(
        web_search_tool=tool,
    )

    context = TaskContext(
        task_id=1,
        task_description=(
            "Find current Python information."
        ),
    )

    result = agent.run(context)

    assert result.status == "failure"

    assert result.data["query"] == (
        "Find current Python information."
    )

    assert result.message == (
        "Web search returned an empty result."
    )


def test_web_research_agent_handles_tool_error():

    class FailingWebSearchTool:

        def run(self, query: str):
            raise RuntimeError(
                "Search service unavailable."
            )

    agent = WebResearchAgent(
        web_search_tool=(
            FailingWebSearchTool()
        ),
    )

    context = TaskContext(
        task_id=1,
        task_description=(
            "Find current Python information."
        ),
    )

    result = agent.run(context)

    assert result.status == "failure"
    assert result.data["query"] == (
        "Find current Python information."
    )

    assert (
        "Search service unavailable."
        in result.message
    )