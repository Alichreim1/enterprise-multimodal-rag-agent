from rag.agent.planner import Planner
from rag.llms.openai_llm import OpenAILLM
from rag.tools.calculator_tool import CalculatorTool
from rag.tools.datetime_tool import DateTimeTool
from rag.tools.tool_manager import ToolManager


def test_planner_selects_datetime_tool():
    llm = OpenAILLM(model="gpt-5.6-luna")

    tool_manager = ToolManager(
        tools=[
            CalculatorTool(),
            DateTimeTool(),
        ]
    )

    planner = Planner(
        llm=llm,
        tool_manager=tool_manager,
    )

    decision = planner.plan("What is the current date and time?")

    assert decision.action == "tool"
    assert decision.tool_name == "datetime"
    assert decision.arguments == {}