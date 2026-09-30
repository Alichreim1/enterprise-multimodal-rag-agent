import os

import pytest

from rag.agents.email_agent import EmailAgent
from rag.models.context import TaskContext
from rag.models.result import AgentResult
from rag.tools.send_email import SendEmailTool


class FakeEmailContentLLM:

    def generate(self, prompt: str) -> str:
        return """
        {
          "subject": "Multi-Agent Integration Test",
          "body": "This email was sent successfully by the Multi-Agent system."
        }
        """


def test_email_agent_with_real_gmail():

    recipient = os.getenv(
        "TEST_EMAIL_RECIPIENT"
    )

    if not recipient:
        pytest.skip(
            "TEST_EMAIL_RECIPIENT is not set."
        )

    email_agent = EmailAgent(
        email_tool=SendEmailTool(),
        llm=FakeEmailContentLLM(),
    )

    research_result = AgentResult(
        status="success",
        data={
            "email": recipient,
        },
        message=(
            "Research completed successfully."
        ),
    )

    context = TaskContext(
        task_id=2,
        task_description=(
            "Send an email with subject "
            "Multi-Agent Integration Test "
            "and body: This email was sent "
            "successfully by the Multi-Agent "
            "system."
        ),
        dependency_results={
            1: research_result,
        },
    )

    result = email_agent.run(context)

    print(
        "\n=== EMAIL RESULT ===\n"
        f"{result}"
    )

    assert result.status == "success"
    assert result.data["recipient"] == recipient

    assert result.data["subject"] == (
        "Multi-Agent Integration Test"
    )

    assert (
        "Email sent successfully"
        in str(result.data["tool_result"])
    )