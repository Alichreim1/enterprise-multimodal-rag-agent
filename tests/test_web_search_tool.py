from rag.tools.web_search_tool import WebSearchTool


def test_web_search_tool_name():
    tool = WebSearchTool()

    assert tool.name == "web_search"


def test_web_search_tool_schema():
    tool = WebSearchTool()

    schema = tool.schema

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "web_search"

    parameters = schema["function"]["parameters"]

    assert parameters["type"] == "object"
    assert "query" in parameters["properties"]
    assert parameters["required"] == ["query"]