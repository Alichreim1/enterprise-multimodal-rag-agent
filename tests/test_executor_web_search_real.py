from rag.agent.executor import Executor
from rag.models.planner import PlannerDecision
from rag.models.task import AgentTask
from rag.tools.calculator_tool import CalculatorTool
from rag.tools.datetime_tool import DateTimeTool
from rag.tools.web_search_tool import WebSearchTool
from rag.tools.tool_manager import ToolManager


def test_executor_web_search_real():

    tool_manager = ToolManager(
        tools=[
            CalculatorTool(),
            DateTimeTool(),
            WebSearchTool(),
        ]
    )

    executor = Executor(
        rag=None,
        tool_manager=tool_manager,
    )

    task = AgentTask(
        id="1",
        description="Search the web for the latest version of Python.",
        task_type="tool",
    )

    decision = PlannerDecision(
        action="tool",
        reason="The task requires web search.",
        tool_name="web_search",
        arguments={
            "query": "What is the latest version of Python?"
        },
    )

    result = executor.execute(
        task=task,
        decision=decision,
    )

    assert isinstance(result, str)
    assert result.strip() != ""