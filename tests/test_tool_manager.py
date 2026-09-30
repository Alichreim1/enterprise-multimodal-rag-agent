from rag.tools.calculator_tool import CalculatorTool
from rag.tools.datetime_tool import DateTimeTool
from rag.tools.web_search_tool import WebSearchTool
from rag.tools.tool_manager import ToolManager
from rag.tools.send_email import SendEmailTool


def test_tool_manager_contains_send_email():

    manager = ToolManager(
        tools=[
            CalculatorTool(),
            DateTimeTool(),
            WebSearchTool(),
            SendEmailTool(),
        ]
    )

    tool = manager.get_tool("send_email")

    assert isinstance(tool, SendEmailTool)


def test_tool_manager_send_email_schema():

    manager = ToolManager(
        tools=[
            CalculatorTool(),
            DateTimeTool(),
            WebSearchTool(),
            SendEmailTool(),
        ]
    )

    schemas = manager.schemas

    names = [
        schema["function"]["name"]
        for schema in schemas
    ]

    assert "send_email" in names