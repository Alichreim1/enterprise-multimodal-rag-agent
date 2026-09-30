import json

from rag.agent.planner import Planner
from rag.tools.calculator_tool import CalculatorTool
from rag.tools.datetime_tool import DateTimeTool
from rag.tools.send_email import SendEmailTool
from rag.tools.tool_manager import ToolManager
from rag.tools.web_search_tool import WebSearchTool


class FakeLLM:
    def __init__(self, response: str):
        self.response = response

    def generate(self, prompt: str) -> str:
        return self.response


def test_planner_send_email():
    llm = FakeLLM(
        json.dumps(
            {
                "action": "tool",
                "tool_name": "send_email",
                "arguments": {
                    "to": "test@example.com",
                    "subject": "Test",
                    "body": "Hello from my RAG Agent.",
                },
            }
        )
    )

    tool_manager = ToolManager(
        tools=[
            CalculatorTool(),
            DateTimeTool(),
            WebSearchTool(),
            SendEmailTool(),
        ]
    )

    planner = Planner(
        llm,
        tool_manager,
    )

    decision = planner.plan(
        'Send an email to test@example.com with subject "Test" '
        'and body "Hello from my RAG Agent."'
    )

    assert decision.action == "tool"
    assert decision.tool_name == "send_email"
    assert decision.arguments == {
        "to": "test@example.com",
        "subject": "Test",
        "body": "Hello from my RAG Agent.",
    }