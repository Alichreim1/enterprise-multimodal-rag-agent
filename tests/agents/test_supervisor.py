from rag.agents.base import BaseAgent
from rag.models.context import TaskContext
from rag.models.result import AgentResult
from rag.agents.agent_registry import AgentRegistry
from rag.agent.state import AgentState
from rag.models.task import AgentTask
from rag.agents.supervisor import Supervisor
import pytest



class FakeAgent(BaseAgent):

    def __init__(
        self,
        result: AgentResult,
    ):
        self.result = result
        self.received_context = None

    @property
    def name(self) -> str:
        return "fake_agent"

    def run(
        self,
        context: TaskContext,
    ) -> AgentResult:

        self.received_context = context

        return self.result


def test_supervisor_executes_current_task():

    expected_result = AgentResult(
        status="success",
        data={
            "email": "ali@example.com",
        },
    )

    agent = FakeAgent(
        result=expected_result,
    )

    registry = AgentRegistry()

    registry.register(
        task_type="retrieve",
        agent=agent,
    )

    supervisor = Supervisor(
        registry=registry,
    )

    task = AgentTask(
        id=1,
        description="Find Ali's email.",
        task_type="retrieve",
    )

    state = AgentState(
        question="Find Ali's email.",
        tasks=[task],
    )

    result = supervisor.execute_current_task(
        state
    )

    assert result is expected_result
    assert state.results[1] is expected_result

    assert task.result is expected_result
    assert task.completed is True
    assert task.failed is False
    assert task.attempts == 1

    assert state.iteration == 1
    assert state.current_task_index == 1

    assert agent.received_context.task_id == 1
    assert (
        agent.received_context.task_description
        == "Find Ali's email."
    )

    assert (
        agent.received_context.dependency_results
        == {}
    )


def test_supervisor_marks_failed_result():

    failed_result = AgentResult(
        status="failure",
        message="Research failed.",
    )

    agent = FakeAgent(
        result=failed_result,
    )

    registry = AgentRegistry()

    registry.register(
        task_type="retrieve",
        agent=agent,
    )

    supervisor = Supervisor(
        registry=registry,
    )

    task = AgentTask(
        id=1,
        description="Find Ali's email.",
        task_type="retrieve",
    )

    state = AgentState(
        question="Find Ali's email.",
        tasks=[task],
    )

    supervisor.execute_current_task(state)

    assert task.completed is False
    assert task.failed is True
    assert state.results[1] is failed_result


def test_supervisor_rejects_unsatisfied_dependencies():

    agent = FakeAgent(
        result=AgentResult(
            status="success",
        ),
    )

    registry = AgentRegistry()

    registry.register(
        task_type="email",
        agent=agent,
    )

    supervisor = Supervisor(
        registry=registry,
    )

    task = AgentTask(
        id=2,
        description="Send Ali an email.",
        task_type="email",
        depends_on=[1],
    )

    state = AgentState(
        question="Find and email Ali.",
        tasks=[task],
    )

    with pytest.raises(
        RuntimeError,
        match="Dependencies are not satisfied",
    ):
        supervisor.execute_current_task(state)

    assert agent.received_context is None
    assert state.results == {}
    assert task.attempts == 0

def test_supervisor_runs_dependent_tasks():

    research_result = AgentResult(
        status="success",
        data={
            "answer": (
                "Ali's email is ali@example.com."
            ),
            "email": "ali@example.com",
        },
    )

    email_result = AgentResult(
        status="success",
        data={
            "recipient": "ali@example.com",
            "subject": "Meeting",
            "body": (
                "The meeting is tomorrow at 5 PM."
            ),
        },
    )

    research_agent = FakeAgent(
        result=research_result,
    )

    email_agent = FakeAgent(
        result=email_result,
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

    supervisor = Supervisor(
        registry=registry,
    )

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

    returned_state = supervisor.run(state)

    assert returned_state is state
    assert state.finished is True
    assert state.current_task_index == 2
    assert state.iteration == 2

    assert research_task.completed is True
    assert email_task.completed is True

    assert state.results[1] is research_result
    assert state.results[2] is email_result

    dependency_results = (
        email_agent
        .received_context
        .dependency_results
    )

    assert dependency_results == {
        1: research_result,
    }


def test_supervisor_stops_after_failed_task():

    failed_research_result = AgentResult(
        status="failure",
        message="Email was not found.",
    )

    research_agent = FakeAgent(
        result=failed_research_result,
    )

    email_agent = FakeAgent(
        result=AgentResult(
            status="success",
        ),
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

    supervisor = Supervisor(
        registry=registry,
    )

    research_task = AgentTask(
        id=1,
        description="Find Ali's email.",
        task_type="retrieve",
    )

    email_task = AgentTask(
        id=2,
        description="Send Ali an email.",
        task_type="email",
        depends_on=[1],
    )

    state = AgentState(
        question="Find and email Ali.",
        tasks=[
            research_task,
            email_task,
        ],
    )

    supervisor.run(state)

    assert state.finished is True
    assert state.current_task_index == 1
    assert state.iteration == 1

    assert research_task.failed is True
    assert email_task.completed is False
    assert email_task.failed is False

    assert email_agent.received_context is None
    assert 1 in state.results
    assert 2 not in state.results
