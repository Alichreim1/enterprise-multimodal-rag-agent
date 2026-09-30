from rag.agent.task_decomposer import TaskDecomposer
from rag.agent.planner import Planner
from rag.agent.executor import Executor
from rag.agent.reflection import Reflection

from rag.models.rag_response import RAGResponse


class FakeLLM:

    def __init__(self):
        self.calls = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)

        # Task decomposition
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

        # Reflection
        if "reflection component" in prompt:

            if "What is Ali's GitHub?" in prompt:
                return """
                {
                    "completed": true,
                    "reason": "The result directly answers the task."
                }
                """

            if "What is 25 * 8?" in prompt:
                return """
                {
                    "completed": true,
                    "reason": "The calculation result completes the task."
                }
                """

        raise AssertionError(
            f"Unexpected prompt:\n{prompt}"
        )


class FakeRAG:

    def __init__(self):
        self.calls = []

    def ask(self, question: str):

        self.calls.append(question)

        return RAGResponse(
            answer="Ali's GitHub is https://github.com/Alichreim1/.",
            search_results=[],
            prompt="fake prompt",
        )


class FakeToolManager:

    def __init__(self):
        self.calls = []

    def execute(
        self,
        name: str,
        arguments: dict,
    ):

        self.calls.append(
            {
                "name": name,
                "arguments": arguments,
            }
        )

        return "25 * 8 = 200"


def test_decomposer_planner_executor_reflection():

    llm = FakeLLM()

    rag = FakeRAG()

    tool_manager = FakeToolManager()

    decomposer = TaskDecomposer(llm)

    planner = Planner(llm)

    executor = Executor(
        rag=rag,
        tool_manager=tool_manager,
    )

    reflection = Reflection(llm)

    # -----------------------------------------
    # 1. Task Decomposition
    # -----------------------------------------

    tasks = decomposer.decompose(
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

    # -----------------------------------------
    # 2. First task → Planner
    # -----------------------------------------

    first_task = tasks[0]

    decision = planner.plan(
        first_task.description
    )

    assert decision.action == "retrieve"

    # -----------------------------------------
    # 3. First task → Executor
    # -----------------------------------------

    result = executor.execute(
        first_task,
        decision,
    )

    assert result.answer == (
        "Ali's GitHub is https://github.com/Alichreim1/."
    )

    assert rag.calls == [
        "What is Ali's GitHub?"
    ]

    # -----------------------------------------
    # 4. First task → store result
    # -----------------------------------------

    first_task.result = result.answer

    # -----------------------------------------
    # 5. First task → Reflection
    # -----------------------------------------

    reflection_decision = reflection.reflect(
        task=first_task.description,
        result=first_task.result,
    )

    assert reflection_decision.completed is True

    first_task.completed = (
        reflection_decision.completed
    )

    first_task.reflection = (
        reflection_decision.reason
    )

    assert first_task.completed is True

    # -----------------------------------------
    # 6. Second task → Planner
    # -----------------------------------------

    second_task = tasks[1]

    decision = planner.plan(
        second_task.description
    )

    assert decision.action == "tool"

    assert decision.tool_name == "calculator"

    assert decision.arguments == {
        "expression": "25 * 8"
    }

    # -----------------------------------------
    # 7. Second task → Executor
    # -----------------------------------------

    result = executor.execute(
        second_task,
        decision,
    )

    assert result == "25 * 8 = 200"

    assert tool_manager.calls == [
        {
            "name": "calculator",
            "arguments": {
                "expression": "25 * 8"
            },
        }
    ]

    # -----------------------------------------
    # 8. Second task → store result
    # -----------------------------------------

    second_task.result = result

    # -----------------------------------------
    # 9. Second task → Reflection
    # -----------------------------------------

    reflection_decision = reflection.reflect(
        task=second_task.description,
        result=second_task.result,
    )

    assert reflection_decision.completed is True

    second_task.completed = (
        reflection_decision.completed
    )

    second_task.reflection = (
        reflection_decision.reason
    )

    assert second_task.completed is True

    # -----------------------------------------
    # Final state
    # -----------------------------------------

    assert first_task.completed is True
    assert second_task.completed is True

    assert first_task.result == (
        "Ali's GitHub is https://github.com/Alichreim1/."
    )

    assert second_task.result == (
        "25 * 8 = 200"
    )