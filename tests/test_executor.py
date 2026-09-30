import pytest

from rag.agent.executor import Executor
from rag.models.planner import PlannerDecision
from rag.models.task import AgentTask


class FakeRAGResponse:

    def __init__(self, answer):
        self.answer = answer

    def __str__(self):
        return self.answer


class FakeRAG:

    def __init__(self):
        self.questions = []

    def ask(self, question):
        self.questions.append(question)

        return FakeRAGResponse(
            "Ali's GitHub is https://github.com/Alichreim1/"
        )


class FakeToolManager:

    def __init__(self):
        self.calls = []

    def execute(self, name, arguments):
        self.calls.append({
            "name": name,
            "arguments": arguments,
        })

        if name == "calculator":
            expression = arguments["expression"]

            if expression == "25 * 8":
                return 200

        raise ValueError(
            f"Unexpected tool call: {name}"
        )


def create_task(
    description,
    task_type,
    result=None,
):
    return AgentTask(
        id=0,
        description=description,
        task_type=task_type,
        result=result,
    )


def test_execute_retrieve():

    rag = FakeRAG()
    tool_manager = FakeToolManager()

    executor = Executor(
        rag=rag,
        tool_manager=tool_manager,
    )

    task = create_task(
        "What is Ali's GitHub?",
        "retrieve",
    )

    decision = PlannerDecision(
        action="retrieve",
        reason="The task requires information retrieval.",
    )

    result = executor.execute(
        task,
        decision,
    )

    assert result.answer == (
        "Ali's GitHub is https://github.com/Alichreim1/"
    )

    assert rag.questions == [
        "What is Ali's GitHub?"
    ]

    assert tool_manager.calls == []


def test_execute_tool():

    rag = FakeRAG()
    tool_manager = FakeToolManager()

    executor = Executor(
        rag=rag,
        tool_manager=tool_manager,
    )

    task = create_task(
        "What is 25 * 8?",
        "tool",
    )

    decision = PlannerDecision(
        action="tool",
        reason="The task requires a calculator.",
        tool_name="calculator",
        arguments={
            "expression": "25 * 8"
        },
    )

    result = executor.execute(
        task,
        decision,
    )

    assert result == 200

    assert tool_manager.calls == [
        {
            "name": "calculator",
            "arguments": {
                "expression": "25 * 8"
            },
        }
    ]

    assert rag.questions == []


def test_execute_finish_with_existing_result():

    rag = FakeRAG()
    tool_manager = FakeToolManager()

    executor = Executor(
        rag=rag,
        tool_manager=tool_manager,
    )

    task = create_task(
        "I prefer Python.",
        "statement",
        result="Preference recorded.",
    )

    decision = PlannerDecision(
        action="finish",
        reason="The task is already completed.",
    )

    result = executor.execute(
        task,
        decision,
    )

    assert result == "Preference recorded."

    assert rag.questions == []
    assert tool_manager.calls == []


def test_execute_finish_without_result():

    rag = FakeRAG()
    tool_manager = FakeToolManager()

    executor = Executor(
        rag=rag,
        tool_manager=tool_manager,
    )

    task = create_task(
        "I prefer Python.",
        "statement",
    )

    decision = PlannerDecision(
        action="finish",
        reason="The task is already completed.",
    )

    result = executor.execute(
        task,
        decision,
    )

    assert result == "I prefer Python."

    assert rag.questions == []
    assert tool_manager.calls == []


def test_invalid_action():

    rag = FakeRAG()
    tool_manager = FakeToolManager()

    executor = Executor(
        rag=rag,
        tool_manager=tool_manager,
    )

    task = create_task(
        "Some task",
        "retrieve",
    )

    decision = PlannerDecision(
        action="invalid",
    )

    with pytest.raises(ValueError):
        executor.execute(
            task,
            decision,
        )