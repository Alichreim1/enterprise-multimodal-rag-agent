from rag.agent.executor import Executor
from rag.models.planner import PlannerDecision
from rag.tools.base_tool import BaseTool
from rag.tools.tool_manager import ToolManager


class FakeRAG:

    def ask(self, question: str):
        return "RAG result"


class FakeSendEmailTool(BaseTool):

    def __init__(self):
        self.received_arguments = None

    @property
    def name(self) -> str:
        return "send_email"

    @property
    def description(self) -> str:
        return "Fake email tool for testing."

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }

    def run(self, **kwargs):
        self.received_arguments = kwargs
        return "fake email sent"


def test_executor_send_email():
    email_tool = FakeSendEmailTool()

    tool_manager = ToolManager(
        tools=[
            email_tool,
        ]
    )

    executor = Executor(
        rag=FakeRAG(),
        tool_manager=tool_manager,
    )

    decision = PlannerDecision(
        action="tool",
        reason="The task requires sending an email.",
        tool_name="send_email",
        arguments={
            "to": "test@example.com",
            "subject": "Test",
            "body": "Hello from my RAG Agent.",
        },
    )

    result = executor.execute(
        task=None,
        decision=decision,
    )

    assert result == "fake email sent"

    assert email_tool.received_arguments == {
        "to": "test@example.com",
        "subject": "Test",
        "body": "Hello from my RAG Agent.",
    }