from rag.agent.context_builder import build_task_context
from rag.models.result import AgentResult
from rag.agent.state import AgentState
from rag.models.task import AgentTask


def test_build_task_context_from_dependencies():

    state = AgentState(
        question="Find Ali's email and send it"
    )

    state.results[1] = AgentResult(
        status="success",
        data={
            "email": "ali@example.com"
        }
    )

    task = AgentTask(
        id=2,
        description="Send email to Ali",
        task_type="tool",
        depends_on=[1],
    )

    context = build_task_context(state, task)

    assert context.task_id == 2

    assert context.task_description == "Send email to Ali"

    assert 1 in context.dependency_results

    assert (
        context.dependency_results[1].data["email"]
        == "ali@example.com"
    )