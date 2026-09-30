from rag.agent.executor import Executor
from rag.agent.state import AgentState
from rag.tools.calculator_tool import CalculatorTool
from rag.tools.tool_manager import ToolManager


class FakeRAGPipeline:

    def retrieve_context(
        self,
        question: str,
    ):

        return []

    def build_prompt(
        self,
        question: str,
        results,
    ):

        return "prompt"

    def generate_answer(
        self,
        prompt: str,
    ):

        return "answer"


def test_executor_executes_calculator_tool():

    calculator = CalculatorTool()

    tool_manager = ToolManager(
        tools=[
            calculator
        ]
    )

    executor = Executor(
        rag_pipeline=FakeRAGPipeline(),
        tool_manager=tool_manager,
    )

    state = AgentState(
        question="What is 25 * 4?"
    )

    state.action = "tool"

    state.tool_name = "calculator"

    state.tool_arguments = {
        "expression": "25 * 4"
    }

    state = executor.execute(
        state
    )

    assert "Tool result:" in (
        state.scratchpad.to_text()
    )

    assert "100" in (
        state.scratchpad.to_text()
    )


def test_executor_requires_tool_manager():

    executor = Executor(
        rag_pipeline=FakeRAGPipeline()
    )

    state = AgentState(
        question="What is 10 + 5?"
    )

    state.action = "tool"

    state.tool_name = "calculator"

    state.tool_arguments = {
        "expression": "10 + 5"
    }

    try:

        executor.execute(
            state
        )

        assert False, (
            "Expected RuntimeError"
        )

    except RuntimeError as error:

        assert "ToolManager" in str(error)


def test_executor_rejects_unknown_tool():

    tool_manager = ToolManager(
        tools=[]
    )

    executor = Executor(
        rag_pipeline=FakeRAGPipeline(),
        tool_manager=tool_manager,
    )

    state = AgentState(
        question="Calculate something"
    )

    state.action = "tool"

    state.tool_name = "unknown_tool"

    state.tool_arguments = {}

    try:

        executor.execute(
            state
        )

        assert False, (
            "Expected ValueError"
        )

    except ValueError as error:

        assert "Unknown tool" in str(error)