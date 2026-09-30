from rag.agent.task_decomposer import (
    TaskDecomposer,
)


class FakeLLM:

    def generate(self, prompt: str):
        return """
        {
          "tasks": [
            {
              "id": 1,
              "description": "Find the latest stable Python version from the web.",
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
              "description": "Write a short report about the latest stable Python version.",
              "task_type": "report",
              "is_statement": false,
              "depends_on": [1]
            },
            {
              "id": 4,
              "description": "Send the generated report to Ali with subject Python Report.",
              "task_type": "email",
              "is_statement": false,
              "depends_on": [2, 3]
            }
          ]
        }
        """


def test_decomposer_creates_report_flow():

    decomposer = TaskDecomposer(
        llm=FakeLLM(),
    )

    tasks = decomposer.decompose(
        (
            "Find the latest stable Python "
            "version, write a report, find "
            "Ali's email, and email the report."
        )
    )

    assert len(tasks) == 4

    web_task = tasks[0]
    retrieve_task = tasks[1]
    report_task = tasks[2]
    email_task = tasks[3]

    assert web_task.task_type == (
        "web_search"
    )

    assert retrieve_task.task_type == (
        "retrieve"
    )

    assert report_task.task_type == "report"

    assert report_task.depends_on == [
        web_task.id
    ]

    assert email_task.task_type == "email"

    assert set(
        email_task.depends_on
    ) == {
        retrieve_task.id,
        report_task.id,
    }