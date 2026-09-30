import pytest

from api.routes.documents import (
    ALLOWED_EXTENSIONS,
    replace_active_research_pipeline,
)


class FakeResearchAgent:

    def __init__(self):
        self.rag = "old_pipeline"


class FakeRegistry:

    def __init__(self, research_agent):
        self.research_agent = research_agent

    def get(self, task_type):

        if task_type == "retrieve":
            return self.research_agent

        return None


class FakeSupervisor:

    def __init__(self, registry):
        self.registry = registry


class FakeRAGAgent:

    def __init__(self, supervisor):
        self.supervisor = supervisor


def test_replace_active_research_pipeline():

    research_agent = FakeResearchAgent()

    registry = FakeRegistry(
        research_agent=research_agent
    )

    supervisor = FakeSupervisor(
        registry=registry
    )

    agent = FakeRAGAgent(
        supervisor=supervisor
    )

    new_pipeline = object()

    replace_active_research_pipeline(
        agent=agent,
        pipeline=new_pipeline,
    )

    assert research_agent.rag is new_pipeline


def test_replace_pipeline_missing_agent():

    registry = FakeRegistry(
        research_agent=None
    )

    supervisor = FakeSupervisor(
        registry=registry
    )

    agent = FakeRAGAgent(
        supervisor=supervisor
    )

    with pytest.raises(
        RuntimeError,
        match="ResearchAgent is not registered",
    ):
        replace_active_research_pipeline(
            agent=agent,
            pipeline=object(),
        )


def test_upload_extensions_include_images():

    assert ".png" in ALLOWED_EXTENSIONS
    assert ".jpg" in ALLOWED_EXTENSIONS
    assert ".jpeg" in ALLOWED_EXTENSIONS
    assert ".webp" in ALLOWED_EXTENSIONS


def test_upload_extensions_keep_documents():

    assert ".pdf" in ALLOWED_EXTENSIONS
    assert ".docx" in ALLOWED_EXTENSIONS
    assert ".txt" in ALLOWED_EXTENSIONS