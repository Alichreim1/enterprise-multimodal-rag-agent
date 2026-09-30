from rag.agent.state import AgentState
from rag.agents.agent_registry import AgentRegistry
from rag.agents.email_agent import EmailAgent
from rag.agents.research_agent import ResearchAgent
from rag.agents.supervisor import Supervisor
from rag.agents.web_research_agent import (
    WebResearchAgent,
)
from rag.models.task import AgentTask


class FakeWebSearchTool:

    def __init__(self):
        self.called_with = None

    def run(self, query: str):
        self.called_with = query

        return (
            "The latest stable Python "
            "version is Python 3.14."
        )


class FakeRAGResponse:

    def __init__(self, answer: str):
        self.answer = answer


class FakeRAG:

    def __init__(self):
        self.called_with = None

    def ask(self, question: str):
        self.called_with = question

        return FakeRAGResponse(
            answer=(
                "Ali's email address is "
                "ali@example.com."
            )
        )


class FakeEmailLLM:

    def __init__(self):
        self.received_prompt = None

    def generate(self, prompt: str):
        self.received_prompt = prompt

        return """
        {
          "subject": "Python Update",
          "body": "The latest stable Python version is Python 3.14."
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


def test_web_research_to_email_flow():

    web_search_tool = FakeWebSearchTool()
    rag = FakeRAG()
    email_llm = FakeEmailLLM()
    email_tool = FakeEmailTool()

    web_research_agent = (
        WebResearchAgent(
            web_search_tool=(
                web_search_tool
            ),
        )
    )

    research_agent = ResearchAgent(
        rag=rag,
    )

    email_agent = EmailAgent(
        email_tool=email_tool,
        llm=email_llm,
    )

    registry = AgentRegistry()

    registry.register(
        task_type="web_search",
        agent=web_research_agent,
    )

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

    web_task = AgentTask(
        id=1,
        description=(
            "Find the latest stable "
            "Python version."
        ),
        task_type="web_search",
        depends_on=[],
    )

    recipient_task = AgentTask(
        id=2,
        description=(
            "Find Ali's email in the "
            "uploaded document."
        ),
        task_type="retrieve",
        depends_on=[],
    )

    email_task = AgentTask(
        id=3,
        description=(
            "Send Ali an email with subject "
            "Python Update containing the "
            "latest stable Python version."
        ),
        task_type="email",
        depends_on=[1, 2],
    )

    state = AgentState(
        question=(
            "Find the latest stable Python "
            "version, find Ali's email, and "
            "email the result to Ali."
        ),
        tasks=[
            web_task,
            recipient_task,
            email_task,
        ],
    )

    supervisor.run(state)

    # -----------------------------------------
    # State
    # -----------------------------------------

    assert state.finished is True
    assert state.current_task_index == 3
    assert state.iteration == 3

    assert web_task.completed is True
    assert recipient_task.completed is True
    assert email_task.completed is True

    # -----------------------------------------
    # WebResearchAgent
    # -----------------------------------------

    assert web_search_tool.called_with == (
        "Find the latest stable "
        "Python version."
    )

    assert (
        state.results[1].data["answer"]
        == (
            "The latest stable Python "
            "version is Python 3.14."
        )
    )

    # -----------------------------------------
    # ResearchAgent
    # -----------------------------------------

    assert rag.called_with == (
        "Find Ali's email in the "
        "uploaded document."
    )

    assert (
        state.results[2].data["email"]
        == "ali@example.com"
    )

    # -----------------------------------------
    # EmailAgent
    # -----------------------------------------

    assert (
        "Python 3.14"
        in email_llm.received_prompt
    )

    assert (
        "ali@example.com"
        in email_llm.received_prompt
    )

    assert email_tool.called_with == {
        "to": "ali@example.com",
        "subject": "Python Update",
        "body": (
            "The latest stable Python "
            "version is Python 3.14."
        ),
    }

    assert state.results[3].status == "success"

    assert state.results[3].data[
        "recipient"
    ] == "ali@example.com"