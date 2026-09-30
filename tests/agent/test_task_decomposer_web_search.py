from rag.agent.task_decomposer import (
    TaskDecomposer,
)


class FakeLLM:

    def generate(self, prompt: str) -> str:
        return """
        {
          "tasks": [
            {
              "id": 1,
              "description": "Find the latest stable Python version.",
              "task_type": "web_search",
              "is_statement": false,
              "depends_on": []
            },
            {
              "id": 2,
              "description": "Find Ali's email in the uploaded document.",
              "task_type": "retrieve",
              "is_statement": false,
              "depends_on": []
            },
            {
              "id": 3,
              "description": "Send Ali an email containing the latest stable Python version.",
              "task_type": "email",
              "is_statement": false,
              "depends_on": [1, 2]
            }
          ]
        }
        """


def test_decomposer_creates_web_search_flow():

    decomposer = TaskDecomposer(
        llm=FakeLLM(),
    )

    tasks = decomposer.decompose(
        (
            "Find the latest stable Python "
            "version, find Ali's email, and "
            "email the result to Ali."
        )
    )

    assert len(tasks) == 3

    web_task = tasks[0]
    research_task = tasks[1]
    email_task = tasks[2]

    assert web_task.id == 1
    assert web_task.task_type == "web_search"
    assert web_task.depends_on == []

    assert research_task.id == 2
    assert research_task.task_type == "retrieve"
    assert research_task.depends_on == []

    assert email_task.id == 3
    assert email_task.task_type == "email"
    assert email_task.depends_on == [1, 2]