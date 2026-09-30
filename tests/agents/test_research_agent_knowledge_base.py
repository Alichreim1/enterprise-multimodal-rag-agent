from rag.agents.research_agent import (
    ResearchAgent,
)


class FakeRAG:
    pass


class FakeKnowledgeBase:
    pass


def test_research_agent_holds_knowledge_base():

    rag = FakeRAG()

    knowledge_base = FakeKnowledgeBase()

    agent = ResearchAgent(
        rag=rag,
        knowledge_base=knowledge_base,
    )

    assert agent.rag is rag

    assert (
        agent.knowledge_base
        is knowledge_base
    )


def test_research_agent_keeps_backward_compatibility():

    rag = FakeRAG()

    agent = ResearchAgent(
        rag=rag,
    )

    assert agent.rag is rag
    assert agent.knowledge_base is None