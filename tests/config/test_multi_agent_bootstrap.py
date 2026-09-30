from config.bootstrap import build_multi_agent_supervisor
from rag.agents.email_agent import EmailAgent
from rag.agents.report_agent import ReportAgent
from rag.agents.research_agent import ResearchAgent
from rag.agents.web_research_agent import WebResearchAgent


class FakeWebSearchTool:
    pass


class FakeRAG:
    pass


class FakeLLM:
    pass


class FakeEmailTool:
    pass


def test_build_multi_agent_supervisor():
    supervisor = build_multi_agent_supervisor(
        rag_pipeline=FakeRAG(),
        reasoning_llm=FakeLLM(),
        email_tool=FakeEmailTool(),
        web_search_tool=FakeWebSearchTool(),
    )

    research_agent = supervisor.registry.get("retrieve")
    email_agent = supervisor.registry.get("email")
    web_research_agent = supervisor.registry.get("web_search")
    report_agent = supervisor.registry.get("report")

    assert isinstance(
        research_agent,
        ResearchAgent,
    )

    assert isinstance(
        email_agent,
        EmailAgent,
    )

    assert isinstance(
        web_research_agent,
        WebResearchAgent,
    )

    assert isinstance(
        report_agent,
        ReportAgent,
    )

    assert research_agent.rag.__class__ is FakeRAG

    assert email_agent.llm.__class__ is FakeLLM
    assert email_agent.email_tool.__class__ is FakeEmailTool

    assert (
        web_research_agent.web_search_tool.__class__
        is FakeWebSearchTool
    )

    assert report_agent.llm.__class__ is FakeLLM
