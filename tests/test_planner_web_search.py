from rag.agent.planner import Planner
from rag.llms.openai_llm import OpenAILLM
from rag.tools.calculator_tool import CalculatorTool
from rag.tools.datetime_tool import DateTimeTool
from rag.tools.web_search_tool import WebSearchTool
from rag.tools.tool_manager import ToolManager


def test_planner_selects_web_search_tool():
    llm = OpenAILLM(model="gpt-5.6-luna")

    tool_manager = ToolManager(
        tools=[
            CalculatorTool(),
            DateTimeTool(),
            WebSearchTool(),
        ]
    )

    planner = Planner(
        llm=llm,
        tool_manager=tool_manager,
    )

    decision = planner.plan(
        "Search the web for the latest information about OpenAI."
    )

    assert decision.action == "tool"
    assert decision.tool_name == "web_search"
    assert decision.arguments["query"] == (
        "latest information about OpenAI"
    )