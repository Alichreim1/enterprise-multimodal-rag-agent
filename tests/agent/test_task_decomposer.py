
from rag.agent.task_decomposer import TaskDecomposer


class FakeLLM:

    def __init__(self, response: str):
        self.response = response

    def generate(self, prompt: str) -> str:
        return self.response


def test_single_retrieve_task():

    response = """
    {
        "tasks": [
            {
                "description": "What is Ali's GitHub?",
                "task_type": "retrieve",
                "is_statement": false
            }
        ]
    }
    """

    decomposer = TaskDecomposer(FakeLLM(response))

    tasks = decomposer.decompose("What is Ali's GitHub?")

    assert len(tasks) == 1

    assert tasks[0].id == 0
    assert tasks[0].description == "What is Ali's GitHub?"
    assert tasks[0].task_type == "retrieve"
    assert tasks[0].is_statement is False


def test_single_tool_task():

    response = """
    {
        "tasks": [
            {
                "description": "What is 25 * 8?",
                "task_type": "tool",
                "is_statement": false
            }
        ]
    }
    """

    decomposer = TaskDecomposer(FakeLLM(response))

    tasks = decomposer.decompose("What is 25 * 8?")

    assert len(tasks) == 1

    assert tasks[0].task_type == "tool"


def test_multiple_tasks():

    response = """
    {
        "tasks": [
            {
                "description": "What is Ali's GitHub?",
                "task_type": "retrieve",
                "is_statement": false
            },
            {
                "description": "What is 25 * 8?",
                "task_type": "tool",
                "is_statement": false
            }
        ]
    }
    """

    decomposer = TaskDecomposer(FakeLLM(response))

    tasks = decomposer.decompose(
        "What is Ali's GitHub and what is 25 * 8?"
    )

    assert len(tasks) == 2

    assert tasks[0].id == 0
    assert tasks[0].task_type == "retrieve"

    assert tasks[1].id == 1
    assert tasks[1].task_type == "tool"


def test_statement():

    response = """
    {
        "tasks": [
            {
                "description": "I prefer Python.",
                "task_type": "statement",
                "is_statement": true
            }
        ]
    }
    """

    decomposer = TaskDecomposer(FakeLLM(response))

    tasks = decomposer.decompose("I prefer Python.")

    assert len(tasks) == 1

    assert tasks[0].task_type == "statement"
    assert tasks[0].is_statement is True


def test_markdown_json():

    response = """
    ```json
    {
        "tasks": [
            {
                "description": "What is 25 * 8?",
                "task_type": "tool",
                "is_statement": false
            }
        ]
    }
    ```
    """

    decomposer = TaskDecomposer(FakeLLM(response))

    tasks = decomposer.decompose("What is 25 * 8?")

    assert len(tasks) == 1
    assert tasks[0].task_type == "tool"


def test_invalid_json():

    response = """
    This is not JSON.
    """

    decomposer = TaskDecomposer(FakeLLM(response))

    try:
        decomposer.decompose("What is 25 * 8?")
        assert False
    except ValueError:
        assert True

