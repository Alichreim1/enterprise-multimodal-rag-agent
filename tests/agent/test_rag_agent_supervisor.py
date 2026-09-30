from rag.agent.rag_agent import RAGAgent
from rag.agents.agent_registry import AgentRegistry
from rag.agents.base import BaseAgent
from rag.agents.supervisor import Supervisor
from rag.models.context import TaskContext
from rag.models.result import AgentResult
from rag.models.task import AgentTask


class FakeMemoryManager:

    def recall_relevant(
        self,
        query: str,
        k: int,
    ):
        return []


class FakeTaskDecomposer:

    def decompose(self, question: str):

        return [
            AgentTask(
                id=1,
                description="Find Ali's email.",
                task_type="retrieve",
            ),
            AgentTask(
                id=2,
                description="Send Ali the email.",
                task_type="email",
                depends_on=[1],
            ),
        ]


class FailingLegacyComponent:

    def __getattr__(self, name):
        raise AssertionError(
            "Legacy planner/executor/reflection "
            "must not handle Multi-Agent tasks."
        )


class FakeResearchAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "research_agent"

    def run(
        self,
        context: TaskContext,
    ) -> AgentResult:

        return AgentResult(
            status="success",
            data={
                "email": "ali@example.com",
            },
        )


class FakeEmailAgent(BaseAgent):

    def __init__(self):
        self.received_context = None

    @property
    def name(self) -> str:
        return "email_agent"

    def run(
        self,
        context: TaskContext,
    ) -> AgentResult:

        self.received_context = context

        recipient = (
            context
            .dependency_results[1]
            .data["email"]
        )

        return AgentResult(
            status="success",
            data={
                "recipient": recipient,
            },
        )


class FakeFinalAnswerGenerator:

    def __init__(self):
        self.received_tasks = None

    def generate(
        self,
        question,
        tasks,
        memories,
    ):
        self.received_tasks = tasks

        return (
            "The email was sent successfully "
            "to ali@example.com."
        )

    def generate_stream(
        self,
        question,
        tasks,
        memories,
    ):
        self.received_tasks = tasks

        yield (
            "The email was sent successfully "
            "to ali@example.com."
        )


def test_rag_agent_uses_supervisor():

    email_agent = FakeEmailAgent()

    registry = AgentRegistry()

    registry.register(
        task_type="retrieve",
        agent=FakeResearchAgent(),
    )

    registry.register(
        task_type="email",
        agent=email_agent,
    )

    supervisor = Supervisor(
        registry=registry,
    )

    final_answer_generator = (
        FakeFinalAnswerGenerator()
    )

    legacy_component = (
        FailingLegacyComponent()
    )

    agent = RAGAgent(
        task_decomposer=FakeTaskDecomposer(),
        planner=legacy_component,
        executor=legacy_component,
        reflection=legacy_component,
        final_answer_generator=(
            final_answer_generator
        ),
        memory_manager=FakeMemoryManager(),
        supervisor=supervisor,
    )

    answer = agent.ask(
        "Find Ali's email and send him "
        "an email."
    )

    assert answer == (
        "The email was sent successfully "
        "to ali@example.com."
    )

    dependency_results = (
        email_agent
        .received_context
        .dependency_results
    )

    assert dependency_results[1].data == {
        "email": "ali@example.com",
    }

    tasks = (
        final_answer_generator
        .received_tasks
    )

    assert tasks[0].completed is True
    assert tasks[1].completed is True

    assert (
        tasks[1].result.data["recipient"]
        == "ali@example.com"
    )

def test_rag_agent_stream_uses_supervisor():

    email_agent = FakeEmailAgent()

    registry = AgentRegistry()

    registry.register(
        task_type="retrieve",
        agent=FakeResearchAgent(),
    )

    registry.register(
        task_type="email",
        agent=email_agent,
    )

    supervisor = Supervisor(
        registry=registry,
    )

    final_answer_generator = (
        FakeFinalAnswerGenerator()
    )

    legacy_component = (
        FailingLegacyComponent()
    )

    agent = RAGAgent(
        task_decomposer=FakeTaskDecomposer(),
        planner=legacy_component,
        executor=legacy_component,
        reflection=legacy_component,
        final_answer_generator=(
            final_answer_generator
        ),
        memory_manager=FakeMemoryManager(),
        supervisor=supervisor,
    )

    events = list(
        agent.stream(
            "Find Ali's email and send "
            "him an email."
        )
    )

    event_types = [
        event["type"]
        for event in events
    ]

    assert event_types.count(
        "execution_started"
    ) == 2

    assert event_types.count(
        "execution_completed"
    ) == 2

    assert event_types.count(
        "task_completed"
    ) == 2

    assert "planning_started" not in event_types
    assert "reflection_started" not in event_types
    assert "agent_error" not in event_types

    assert event_types[-1] == "agent_finished"

    multi_agent_events = [
        event
        for event in events
        if (
            event["type"]
            == "execution_completed"
        )
    ]

    assert (
        multi_agent_events[0]
        ["metadata"]["mode"]
        == "multi_agent"
    )

    assert (
        multi_agent_events[0]
        ["metadata"]["data"]["email"]
        == "ali@example.com"
    )

    assert (
        multi_agent_events[1]
        ["metadata"]["data"]["recipient"]
        == "ali@example.com"
    )

    final_answer_event = next(
        event
        for event in events
        if event["type"] == "final_answer"
    )

    assert final_answer_event["result"] == (
        "The email was sent successfully "
        "to ali@example.com."
    )    