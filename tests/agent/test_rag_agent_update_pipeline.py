from rag.agent.rag_agent import RAGAgent
from rag.agents.agent_registry import AgentRegistry
from rag.agents.research_agent import ResearchAgent
from rag.agents.supervisor import Supervisor


class FakeLegacyExecutor:

    def __init__(self):
        self.rag = None


class FakeRAG:
    pass


class Placeholder:
    pass


def test_update_rag_pipeline_updates_both_flows():

    research_agent = ResearchAgent(
        rag=None,
    )

    registry = AgentRegistry()

    registry.register(
        task_type="retrieve",
        agent=research_agent,
    )

    supervisor = Supervisor(
        registry=registry,
    )

    executor = FakeLegacyExecutor()

    agent = RAGAgent(
        task_decomposer=Placeholder(),
        planner=Placeholder(),
        executor=executor,
        reflection=Placeholder(),
        final_answer_generator=Placeholder(),
        memory_manager=Placeholder(),
        supervisor=supervisor,
    )

    new_rag_pipeline = FakeRAG()

    agent.update_rag_pipeline(
        new_rag_pipeline
    )

    assert (
        agent.executor.rag
        is new_rag_pipeline
    )

    assert (
        research_agent.rag
        is new_rag_pipeline
    )