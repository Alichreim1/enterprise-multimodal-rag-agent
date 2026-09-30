from rag.agent.executor import Executor
from rag.models.planner import PlannerDecision
from rag.models.task import AgentTask
from rag.tools.calculator_tool import CalculatorTool
from rag.tools.datetime_tool import DateTimeTool
from rag.tools.web_search_tool import WebSearchTool
from rag.tools.tool_manager import ToolManager


def test_executor_executes_web_search_tool():
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
        description="Search the web for the latest information about OpenAI.",
        task_type="tool",
    )

    decision = PlannerDecision(
        action="tool",
        reason="The task requires web search.",
        tool_name="web_search",
        arguments={
            "query": "latest information about OpenAI",
        },
    )

    try:
        executor.execute(
            task=task,
            decision=decision,
        )
        assert False, "Expected NotImplementedError"
    except NotImplementedError:
        assert True