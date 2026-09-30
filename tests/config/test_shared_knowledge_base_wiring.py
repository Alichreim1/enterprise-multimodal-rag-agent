from rag.agents.research_agent import (
    ResearchAgent,
)


class FakeKnowledgeBase:
    pass


class FakePipeline:
    pass


def test_research_agent_can_hold_shared_pipeline():

    knowledge_base = FakeKnowledgeBase()
    pipeline = FakePipeline()

    research_agent = ResearchAgent(
        rag=pipeline,
        knowledge_base=knowledge_base,
    )

    assert research_agent.rag is pipeline

    assert (
        research_agent.knowledge_base
        is knowledge_base
    )