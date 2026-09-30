import pytest

from rag.agents.web_research_agent import (
    WebResearchAgent,
)
from rag.models.context import TaskContext
from rag.tools.web_search_tool import (
    WebSearchTool,
)


@pytest.mark.integration
def test_web_research_agent_real():

    agent = WebResearchAgent(
        web_search_tool=WebSearchTool(
            model="gpt-5.6-luna",
        ),
    )

    context = TaskContext(
        task_id=1,
        task_description=(
            "What is the latest stable "
            "version of Python?"
        ),
    )

    result = agent.run(context)

    print(
        "\n=== WEB RESEARCH RESULT ==="
    )

    print(result)

    assert result.status == "success"

    assert result.data["query"] == (
        "What is the latest stable "
        "version of Python?"
    )

    answer = result.data["answer"]

    assert isinstance(answer, str)
    assert answer.strip()

    normalized_answer = answer.lower()

    assert "python" in normalized_answer

    assert result.message == (
        "Web research completed "
        "successfully."
    )