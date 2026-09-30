from rag.agents.research_agent import ResearchAgent
from rag.agents.email_agent import EmailAgent
from rag.models.context import TaskContext


class FakeRAG:
    def ask(self, question):
        class Response:
            answer = "ali@example.com"

        return Response()


class FakeLLM:
    def generate(self, prompt):
        return """
        {
            "subject": "Meeting",
            "body": "The meeting is tomorrow at 5 PM."
        }
        """


class FakeEmailTool:
    def __init__(self):
        self.called_with = None

    def run(self, to, subject, body):
        self.called_with = {
            "to": to,
            "subject": subject,
            "body": body,
        }

        return "Email sent successfully."


def test_research_to_email_flow():
    # -----------------------------------------
    # Arrange
    # -----------------------------------------

    research_agent = ResearchAgent(FakeRAG())

    research_context = TaskContext(
        task_id=1,
        task_description="Find Ali's email",
    )

    email_tool = FakeEmailTool()

    email_agent = EmailAgent(
        email_tool=email_tool,
        llm=FakeLLM(),
    )

    # -----------------------------------------
    # Act: Research task
    # -----------------------------------------

    research_result = research_agent.run(
        research_context
    )

    # -----------------------------------------
    # Assert: Research result
    # -----------------------------------------

    assert research_result.status == "success"

    assert research_result.data["email"] == (
        "ali@example.com"
    )

    # -----------------------------------------
    # Arrange: Email task
    # -----------------------------------------

    email_context = TaskContext(
        task_id=2,
        task_description=(
            "Send Ali an email about the meeting."
        ),
        dependency_results={
            1: research_result,
        },
    )

    # -----------------------------------------
    # Act: Email task
    # -----------------------------------------

    email_result = email_agent.run(
        email_context
    )

    # -----------------------------------------
    # Assert: Email result
    # -----------------------------------------

    assert email_result.status == "success"

    assert email_tool.called_with["to"] == (
        "ali@example.com"
    )

    assert email_tool.called_with["subject"] == (
        "Meeting"
    )
