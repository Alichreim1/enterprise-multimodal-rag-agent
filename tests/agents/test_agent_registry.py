from rag.agents.agent_registry import AgentRegistry
from rag.agents.email_agent import EmailAgent
from rag.agents.research_agent import ResearchAgent
from rag.agents.report_agent import ReportAgent
from rag.agents.test_agent import TestAgent
from rag.agents.web_research_agent import WebResearchAgent


class FakeRAG:
    def ask(self, question: str):
        raise NotImplementedError


class FakeLLM:
    def generate(self, prompt: str):
        raise NotImplementedError


class FakeEmailTool:
    def run(
        self,
        to: str,
        subject: str,
        body: str,
    ):
        raise NotImplementedError


class FakeWebTool:
    def run(self, query: str):
        return "result"


class FakeReportLLM:
    def generate(self, prompt: str):
        return """
        {
          "title": "Test",
          "content": "Test content."
        }
        """


def test_registry_routes_to_correct_agents():
    research_agent = ResearchAgent(
        rag=FakeRAG(),
    )

    email_agent = EmailAgent(
        email_tool=FakeEmailTool(),
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

    assert registry.get("retrieve") is research_agent
    assert registry.get("email") is email_agent

    assert registry.get("retrieve").name == "research_agent"
    assert registry.get("email").name == "email_agent"


def test_registry_checks_supported_task_type():
    registry = AgentRegistry()
    agent = TestAgent()

    registry.register(
        task_type="retrieve",
        agent=agent,
    )

    assert registry.has("retrieve") is True
    assert registry.has("email") is False


def test_registry_routes_web_search_tasks():
    web_agent = WebResearchAgent(
        web_search_tool=FakeWebTool(),
    )

    registry = AgentRegistry()

    registry.register(
        task_type="web_search",
        agent=web_agent,
    )

    assert registry.get("web_search") is web_agent
    assert registry.has("web_search") is True


def test_registry_routes_report_tasks():
    report_agent = ReportAgent(
        llm=FakeReportLLM(),
    )

    registry = AgentRegistry()

    registry.register(
        task_type="report",
        agent=report_agent,
    )

    assert registry.get("report") is report_agent
    assert registry.has("report") is True
