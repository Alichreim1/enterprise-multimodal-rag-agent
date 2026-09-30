from rag.agent.state import AgentState
from rag.agent.task_decomposer import TaskDecomposer
from rag.agents.agent_registry import AgentRegistry
from rag.agents.email_agent import EmailAgent
from rag.agents.research_agent import ResearchAgent
from rag.agents.supervisor import Supervisor


class FakeDecomposerLLM:

    def generate(self, prompt: str) -> str:
        return """
        {
          "tasks": [
            {
              "id": 1,
              "description": "Find Ali's email.",
              "task_type": "retrieve",
              "is_statement": false,
              "depends_on": []
            },
            {
              "id": 2,
              "description": "Send Ali an email with subject Meeting and body: The meeting is tomorrow at 5 PM.",
              "task_type": "email",
              "is_statement": false,
              "depends_on": [1]
            }
          ]
        }
        """


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


class FakeEmailLLM:

    def generate(self, prompt: str) -> str:
        return """
        {
          "subject": "Meeting",
          "body": "The meeting is tomorrow at 5 PM."
        }
        """


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


def test_decomposer_to_supervisor_flow():

    question = (
        "Find Ali's email and send him an email "
        "with subject Meeting and body: "
        "The meeting is tomorrow at 5 PM."
    )

    # Task decomposition
    task_decomposer = TaskDecomposer(
        llm=FakeDecomposerLLM(),
    )

    tasks = task_decomposer.decompose(
        question
    )

    # Agents
    email_tool = FakeEmailTool()

    research_agent = ResearchAgent(
        rag=FakeRAG(),
    )

    email_agent = EmailAgent(
        email_tool=email_tool,
        llm=FakeEmailLLM(),
    )

    # Registry
    registry = AgentRegistry()

    registry.register(
        task_type="retrieve",
        agent=research_agent,
    )

    registry.register(
        task_type="email",
        agent=email_agent,
    )

    # Supervisor
    supervisor = Supervisor(
        registry=registry,
    )

    state = AgentState(
        question=question,
        tasks=tasks,
    )

    supervisor.run(state)

    # Decomposition
    assert len(state.tasks) == 2
    assert state.tasks[0].depends_on == []
    assert state.tasks[1].depends_on == [1]

    # Execution state
    assert state.finished is True
    assert state.current_task_index == 2
    assert state.iteration == 2

    # Research output
    research_result = state.results[1]

    assert research_result.status == "success"
    assert (
        research_result.data["email"]
        == "ali@example.com"
    )

    # Email output
    email_result = state.results[2]

    assert email_result.status == "success"
    assert (
        email_result.data["recipient"]
        == "ali@example.com"
    )

    # Actual tool call
    assert email_tool.called_with == {
        "to": "ali@example.com",
        "subject": "Meeting",
        "body": (
            "The meeting is tomorrow at 5 PM."
        ),
    }