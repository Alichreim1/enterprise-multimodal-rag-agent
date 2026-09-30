from rag.tools.send_email import SendEmailTool


def test_send_email_tool_name():

    tool = SendEmailTool()

    assert tool.name == "send_email"


def test_send_email_tool_description():

    tool = SendEmailTool()

    assert tool.description == (
        "Send an email using Gmail."
    )


def test_send_email_tool_schema():

    tool = SendEmailTool()

    schema = tool.schema

    assert schema["type"] == "function"

    function = schema["function"]

    assert function["name"] == "send_email"

    parameters = function["parameters"]

    assert parameters["type"] == "object"

    assert "to" in parameters["properties"]
    assert "subject" in parameters["properties"]
    assert "body" in parameters["properties"]

    assert parameters["required"] == [
        "to",
        "subject",
        "body",
    ]