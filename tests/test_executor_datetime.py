from rag.agent.executor import Executor
from rag.models.planner import PlannerDecision
from rag.models.task import AgentTask
from rag.tools.calculator_tool import CalculatorTool
from rag.tools.datetime_tool import DateTimeTool
from rag.tools.tool_manager import ToolManager


def test_executor_executes_datetime_tool():
    tool_manager = ToolManager(
        tools=[
            CalculatorTool(),
            DateTimeTool(),
        ]
    )

    executor = Executor(
        rag=None,
        tool_manager=tool_manager,
    )

    task = AgentTask(
        id="1",
        description="What is the current date and time?",
        task_type="tool",
    )

    decision = PlannerDecision(
        action="tool",
        reason="The task requires the datetime tool.",
        tool_name="datetime",
        arguments={},
    )

    result = executor.execute(
        task=task,
        decision=decision,
    )

    assert isinstance(result, str)
    assert "T" in result