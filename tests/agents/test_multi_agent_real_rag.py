from config.bootstrap import build_rag_pipeline
from rag.agent.state import AgentState
from rag.agents.agent_registry import AgentRegistry
from rag.agents.email_agent import EmailAgent
from rag.agents.research_agent import ResearchAgent
from rag.agents.supervisor import Supervisor
from rag.models.task import AgentTask


class ContextAwareRAGLLM:

    def generate(self, prompt: str) -> str:

        assert "ali@example.com" in prompt

        return (
            "Ali's email address is "
            "ali@example.com."
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


def test_multi_agent_with_real_rag(
    tmp_path,
):

    # -----------------------------------------
    # Real document
    # -----------------------------------------

    document_path = (
        tmp_path
        / "ali_profile.txt"
    )

    document_path.write_text(
        (
            "Ali Chreim is an AI engineer.\n"
            "Ali's email address is "
            "ali@example.com.\n"
            "Ali works on RAG systems."
        ),
        encoding="utf-8",
    )

    # -----------------------------------------
    # Real RAG pipeline
    # -----------------------------------------

    rag_pipeline = build_rag_pipeline(
        file_path=document_path,
        answer_llm=ContextAwareRAGLLM(),
    )

    # -----------------------------------------
    # Real agents
    # -----------------------------------------

    email_tool = FakeEmailTool()

    research_agent = ResearchAgent(
        rag=rag_pipeline,
    )

    email_agent = EmailAgent(
        email_tool=email_tool,
        llm=FakeEmailLLM(),
    )

    # -----------------------------------------
    # Registry + Supervisor
    # -----------------------------------------

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

    # -----------------------------------------
    # Tasks
    # -----------------------------------------

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
            "the meeting email."
        ),
        tasks=[
            research_task,
            email_task,
        ],
    )

    # -----------------------------------------
    # Execute
    # -----------------------------------------

    supervisor.run(state)

    # -----------------------------------------
    # Assertions
    # -----------------------------------------

    assert state.finished is True
    assert state.current_task_index == 2

    assert (
        state.results[1].data["email"]
        == "ali@example.com"
    )

    assert state.results[2].status == "success"

    assert email_tool.called_with == {
        "to": "ali@example.com",
        "subject": "Meeting",
        "body": (
            "The meeting is tomorrow at 5 PM."
        ),
    }