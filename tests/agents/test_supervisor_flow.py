from rag.agent.state import AgentState
from rag.agents.agent_registry import AgentRegistry
from rag.agents.email_agent import EmailAgent
from rag.agents.research_agent import ResearchAgent
from rag.agents.supervisor import Supervisor
from rag.models.task import AgentTask


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


class FakeLLM:

    def generate(self, prompt: str):
        return (
            "{"
            '"subject": "Meeting", '
            '"body": "The meeting is tomorrow '
            'at 5 PM."'
            "}"
        )


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


def test_research_to_email_supervisor_flow():

    email_tool = FakeEmailTool()

    research_agent = ResearchAgent(
        rag=FakeRAG(),
    )

    email_agent = EmailAgent(
        email_tool=email_tool,
        llm=FakeLLM(),
    )

    registry = AgentRegistry()

    registry.register(
        task_type="retrieve",
        agent=research_agent,
    )

    registry.register(
        task_type="email",
        agent=email_agent,
    )

    supervisor = Supervisor(
        registry=registry,
    )

    research_task = AgentTask(
        id=1,
        description="Find Ali's email.",
        task_type="retrieve",
        depends_on=[],
    )

    email_task = AgentTask(
        id=2,
        description=(
            "Send Ali an email with subject "
            "Meeting and body: The meeting is "
            "tomorrow at 5 PM."
        ),
        task_type="email",
        depends_on=[1],
    )

    state = AgentState(
        question=(
            "Find Ali's email and send him "
            "an email with subject Meeting "
            "and body: The meeting is tomorrow "
            "at 5 PM."
        ),
        tasks=[
            research_task,
            email_task,
        ],
    )

    supervisor.run(state)

    assert state.finished is True
    assert state.current_task_index == 2

    # Research result
    assert state.results[1].status == "success"
    assert (
        state.results[1].data["email"]
        == "ali@example.com"
    )

    assert email_task.result.status == "success"

   

    assert email_task.result.status == "success"

    # Email tool call
    assert email_tool.called_with == {
        "to": "ali@example.com",
        "subject": "Meeting",
        "body": (
            "The meeting is tomorrow at 5 PM."
        ),
    }

    # Email agent result
    assert state.results[2].status == "success"

    assert state.results[2].data == {
        "recipient": "ali@example.com",
        "subject": "Meeting",
        "body": (
            "The meeting is tomorrow at 5 PM."
        ),
        "tool_result": (
            "Email sent successfully."
        ),
    }