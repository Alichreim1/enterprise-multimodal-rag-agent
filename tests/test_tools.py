from rag.tools.calculator_tool import CalculatorTool
from rag.tools.tool_manager import ToolManager


def test_calculator_tool():

    tool = CalculatorTool()

    result = tool.run(
        expression="10 * 5"
    )

    assert result == 50


def test_tool_manager_registers_tools():

    calculator = CalculatorTool()

    manager = ToolManager(
        tools=[
            calculator
        ]
    )

    tool = manager.get_tool(
        "calculator"
    )

    assert tool is calculator


def test_tool_manager_executes_tool():

    calculator = CalculatorTool()

    manager = ToolManager(
        tools=[
            calculator
        ]
    )

    result = manager.execute(
        name="calculator",
        arguments={
            "expression": "20 + 30"
        },
    )

    assert result == 50


def test_tool_manager_rejects_unknown_tool():

    manager = ToolManager(
        tools=[]
    )

    try:

        manager.get_tool(
            "calculator"
        )

        assert False

    except ValueError as error:

        assert "Unknown tool" in str(error)