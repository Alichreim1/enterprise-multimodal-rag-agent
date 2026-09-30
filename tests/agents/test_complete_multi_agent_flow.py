from rag.agent.state import AgentState
from rag.agents.agent_registry import AgentRegistry
from rag.agents.email_agent import EmailAgent
from rag.agents.report_agent import ReportAgent
from rag.agents.research_agent import ResearchAgent
from rag.agents.supervisor import Supervisor
from rag.agents.web_research_agent import (
    WebResearchAgent,
)
from rag.models.task import AgentTask


class FakeWebSearchTool:

    def run(self, query: str):
        return (
            "Python 3.14.7 is the latest stable "
            "Python version, released on "
            "August 5, 2026."
        )


class FakeRAGResponse:

    def __init__(self, answer: str):
        self.answer = answer


class FakeRAG:

    def ask(self, question: str):
        return FakeRAGResponse(
            answer="Ali's email is ali@example.com."
        )


class FakeReportLLM:

    def __init__(self):
        self.received_prompt = None

    def generate(self, prompt: str):
        self.received_prompt = prompt

        return """
        {
          "title": "Python Report",
          "content": "Python 3.14.7 is the latest stable Python version. It was released on August 5, 2026."
        }
        """


class FakeEmailLLM:

    def __init__(self):
        self.received_prompt = None

    def generate(self, prompt: str):
        self.received_prompt = prompt

        return """
        {
          "subject": "Python Report",
          "body": "Python 3.14.7 is the latest stable Python version. It was released on August 5, 2026."
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


def test_complete_multi_agent_flow():

    report_llm = FakeReportLLM()
    email_llm = FakeEmailLLM()
    email_tool = FakeEmailTool()

    web_agent = WebResearchAgent(
        web_search_tool=(
            FakeWebSearchTool()
        ),
    )

    research_agent = ResearchAgent(
        rag=FakeRAG(),
    )

    report_agent = ReportAgent(
        llm=report_llm,
    )

    email_agent = EmailAgent(
        email_tool=email_tool,
        llm=email_llm,
    )

    registry = AgentRegistry()

    registry.register(
        task_type="web_search",
        agent=web_agent,
    )

    registry.register(
        task_type="retrieve",
        agent=research_agent,
    )

    registry.register(
        task_type="report",
        agent=report_agent,
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
            "Find the latest stable Python "
            "version from the web."
        ),
        task_type="web_search",
        depends_on=[],
    )

    retrieve_task = AgentTask(
        id=2,
        description=(
            "Find Ali's email in the "
            "uploaded document."
        ),
        task_type="retrieve",
        depends_on=[],
    )

    report_task = AgentTask(
        id=3,
        description=(
            "Write a short report about the "
            "latest stable Python version."
        ),
        task_type="report",
        depends_on=[1],
    )

    email_task = AgentTask(
        id=4,
        description=(
            "Send the generated report to "
            "Ali with subject Python Report."
        ),
        task_type="email",
        depends_on=[2, 3],
    )

    state = AgentState(
        question=(
            "Find the latest Python version, "
            "write a report, find Ali's email, "
            "and email the report to Ali."
        ),
        tasks=[
            web_task,
            retrieve_task,
            report_task,
            email_task,
        ],
    )

    supervisor.run(state)

    # -----------------------------------------
    # Complete State
    # -----------------------------------------

    assert state.finished is True
    assert state.current_task_index == 4
    assert state.iteration == 4

    assert all(
        task.completed
        for task in state.tasks
    )

    # -----------------------------------------
    # Web Result
    # -----------------------------------------

    assert state.results[1].status == (
        "success"
    )

    assert (
        "Python 3.14.7"
        in state.results[1].data["answer"]
    )

    # -----------------------------------------
    # RAG Result
    # -----------------------------------------

    assert state.results[2].data[
        "email"
    ] == "ali@example.com"

    # -----------------------------------------
    # Report Result
    # -----------------------------------------

    assert state.results[3].data[
        "title"
    ] == "Python Report"

    assert (
        "Python 3.14.7"
        in state.results[3].data["content"]
    )

    assert (
        "Python 3.14.7"
        in report_llm.received_prompt
    )

    # -----------------------------------------
    # Email Result
    # -----------------------------------------

    assert (
        "Python Report"
        in email_llm.received_prompt
    )

    assert (
        "Python 3.14.7"
        in email_llm.received_prompt
    )

    assert email_tool.called_with == {
        "to": "ali@example.com",
        "subject": "Python Report",
        "body": (
            "Python 3.14.7 is the latest "
            "stable Python version. It was "
            "released on August 5, 2026."
        ),
    }

    assert state.results[4].status == (
        "success"
    )

    assert state.results[4].data[
        "recipient"
    ] == "ali@example.com"