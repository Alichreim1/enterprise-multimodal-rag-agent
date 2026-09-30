from rag.agent.executor import Executor
from rag.agent.planner import Planner
from rag.agent.task_decomposer import TaskDecomposer
from rag.models.planner import PlannerDecision
from rag.models.task import AgentTask


class FakeLLM:

    def __init__(self):
        self.calls = []

    def generate(self, prompt: str) -> str:

        self.calls.append(prompt)

        # TaskDecomposer
        if "task decomposition component" in prompt:

            return """
            {
                "tasks": [
                    {
                        "description": "What is Ali's GitHub?",
                        "task_type": "retrieve",
                        "is_statement": false
                    },
                    {
                        "description": "What is 25 * 8?",
                        "task_type": "tool",
                        "is_statement": false
                    }
                ]
            }
            """

        # Planner
        if "planning component" in prompt:

            if "What is Ali's GitHub?" in prompt:
                return """
                {
                    "action": "retrieve",
                    "reason": "The task requires information retrieval.",
                    "tool_name": null,
                    "arguments": {}
                }
                """

            if "What is 25 * 8?" in prompt:
                return """
                {
                    "action": "tool",
                    "reason": "The task requires a calculator.",
                    "tool_name": "calculator",
                    "arguments": {
                        "expression": "25 * 8"
                    }
                }
                """

        raise ValueError(
            "Unexpected prompt received by FakeLLM."
        )


class FakeRAGResponse:

    def __init__(self, answer):
        self.answer = answer


class FakeRAG:

    def __init__(self):
        self.calls = []

    def ask(self, question):

        self.calls.append(question)

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
            return 200

        raise ValueError(
            f"Unexpected tool: {name}"
        )


def test_task_decomposer_planner_executor():

    llm = FakeLLM()

    task_decomposer = TaskDecomposer(
        llm=llm
    )

    planner = Planner(
        llm=llm
    )

    rag = FakeRAG()

    tool_manager = FakeToolManager()

    executor = Executor(
        rag=rag,
        tool_manager=tool_manager,
    )

    # ----------------------------------------
    # 1. TASK DECOMPOSITION
    # ----------------------------------------

    tasks = task_decomposer.decompose(
        "What is Ali's GitHub and what is 25 * 8?"
    )

    assert len(tasks) == 2

    assert tasks[0].description == (
        "What is Ali's GitHub?"
    )

    assert tasks[0].task_type == "retrieve"

    assert tasks[1].description == (
        "What is 25 * 8?"
    )

    assert tasks[1].task_type == "tool"

    # ----------------------------------------
    # 2. FIRST TASK → PLANNER
    # ----------------------------------------

    first_task = tasks[0]

    first_decision = planner.plan(
        first_task.description
    )

    assert first_decision.action == "retrieve"

    # ----------------------------------------
    # 3. FIRST TASK → EXECUTOR
    # ----------------------------------------

    first_result = executor.execute(
        first_task,
        first_decision,
    )

    assert first_result.answer == (
        "Ali's GitHub is https://github.com/Alichreim1/"
    )

    assert rag.calls == [
        "What is Ali's GitHub?"
    ]

    # ----------------------------------------
    # 4. SECOND TASK → PLANNER
    # ----------------------------------------

    second_task = tasks[1]

    second_decision = planner.plan(
        second_task.description
    )

    assert second_decision.action == "tool"

    assert second_decision.tool_name == "calculator"

    assert second_decision.arguments == {
        "expression": "25 * 8"
    }

    # ----------------------------------------
    # 5. SECOND TASK → EXECUTOR
    # ----------------------------------------

    second_result = executor.execute(
        second_task,
        second_decision,
    )

    assert second_result == 200

    assert tool_manager.calls == [
        {
            "name": "calculator",
            "arguments": {
                "expression": "25 * 8"
            }
        }
    ]