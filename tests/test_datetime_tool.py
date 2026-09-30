from rag.tools.datetime_tool import DateTimeTool


def test_datetime_tool_name():
    tool = DateTimeTool()

    assert tool.name == "datetime"


def test_datetime_tool_schema():
    tool = DateTimeTool()

    schema = tool.schema

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "datetime"
    assert schema["function"]["parameters"]["type"] == "object"


def test_datetime_tool_run():
    tool = DateTimeTool()

    result = tool.run()

    assert isinstance(result, str)
    assert "T" in result