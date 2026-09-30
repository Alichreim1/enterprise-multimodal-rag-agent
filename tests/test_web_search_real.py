import pytest

from rag.tools.web_search_tool import WebSearchTool


@pytest.mark.integration
def test_web_search_real():
    tool = WebSearchTool()

    result = tool.run(
        query="What is the latest version of Python?"
    )

    assert isinstance(result, str)
    assert result.strip() != ""