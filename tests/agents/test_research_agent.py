from rag.agents.research_agent import ResearchAgent
from rag.models.context import TaskContext


class FakeRAGResponse:

    def __init__(self, answer: str):
        self.answer = answer


class FakeRAG:

    def ask(self, question: str):
        return FakeRAGResponse(
            answer=(
                "Ali's email address is "
                "ali@example.com."
            )
        )


def test_research_agent_extracts_email():

    agent = ResearchAgent(
        rag=FakeRAG(),
    )

    context = TaskContext(
        task_id=1,
        task_description="Find Ali's email.",
    )

    result = agent.run(context)

    assert result.status == "success"

    assert result.data["answer"] == (
        "Ali's email address is "
        "ali@example.com."
    )

    assert result.data["email"] == (
        "ali@example.com"
    )

    assert result.message == (
        "Research completed successfully."
    )