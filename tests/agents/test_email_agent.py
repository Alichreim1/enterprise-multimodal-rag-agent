from rag.agents.email_agent import EmailAgent
from rag.models.context import TaskContext
from rag.models.result import AgentResult


class FakeEmailTool:
    def __init__(self):
        self.called_with = None

    def run(
        self,
        to: str,
        subject: str,
        body: str,
    ):
        self.called_with = {
            "to": to,
            "subject": subject,
            "body": body,
        }

        return "Email sent successfully."


class FakeLLM:
    def generate(self, prompt: str) -> str:
        return """
        {
            "subject": "Meeting",
            "body": "The meeting is tomorrow at 5 PM."
        }
        """


def test_email_agent_calls_email_tool():
    # -----------------------------------------
    # Arrange
    # -----------------------------------------

    email_tool = FakeEmailTool()
    llm = FakeLLM()

    agent = EmailAgent(
        email_tool=email_tool,
        llm=llm,
    )

    research_result = AgentResult(
        status="success",
        data={
            "email": "ali@example.com",
        },
    )

    context = TaskContext(
        task_id=2,
        task_description="Hello Ali, welcome to the project.",
        dependency_results={
            1: research_result,
        },
    )

    # -----------------------------------------
    # Act
    # -----------------------------------------

    result = agent.run(context)

    # -----------------------------------------
    # Assert: Agent result
    # -----------------------------------------

    assert result.status == "success"

    assert result.data["recipient"] == "ali@example.com"

    # -----------------------------------------
    # Assert: Email tool invocation
    # -----------------------------------------

    assert email_tool.called_with["to"] == "ali@example.com"

    assert email_tool.called_with["subject"] == "Meeting"

    assert email_tool.called_with["body"] == (
        "The meeting is tomorrow at 5 PM."
    )


def test_email_agent_uses_multiple_dependencies():

    class RecordingLLM:

        def __init__(self):
            self.prompt = None

        def generate(self, prompt: str):
            self.prompt = prompt

            return """
            {
              "subject": "Python Update",
              "body": "The latest stable Python version is Python 3.14."
            }
            """

    email_tool = FakeEmailTool()
    llm = RecordingLLM()

    agent = EmailAgent(
        email_tool=email_tool,
        llm=llm,
    )

    web_result = AgentResult(
        status="success",
        data={
            "query": (
                "Find the latest stable "
                "Python version."
            ),
            "answer": (
                "The latest stable Python "
                "version is Python 3.14."
            ),
        },
    )

    recipient_result = AgentResult(
        status="success",
        data={
            "email": "ali@example.com",
        },
    )

    context = TaskContext(
        task_id=3,
        task_description=(
            "Send Ali an email with subject "
            "Python Update containing the "
            "latest stable Python version."
        ),
        dependency_results={
            1: web_result,
            2: recipient_result,
        },
    )

    result = agent.run(context)

    assert result.status == "success"

    assert email_tool.called_with == {
        "to": "ali@example.com",
        "subject": "Python Update",
        "body": (
            "The latest stable Python "
            "version is Python 3.14."
        ),
    }

    assert "Python 3.14" in llm.prompt
    assert "ali@example.com" in llm.prompt