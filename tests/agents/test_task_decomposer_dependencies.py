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
              "description": "Find Ali's email.",
              "task_type": "retrieve",
              "is_statement": false,
              "depends_on": []
            },
            {
              "id": 2,
              "description": "Send Ali an email with subject Meeting and body: The meeting is tomorrow at 5 PM.",
              "task_type": "email",
              "is_statement": false,
              "depends_on": [1]
            }
          ]
        }
        """


def test_decomposer_creates_email_dependency():

    decomposer = TaskDecomposer(
        llm=FakeLLM(),
    )

    tasks = decomposer.decompose(
        "Find Ali's email and send him "
        "an email with subject Meeting."
    )

    assert len(tasks) == 2

    research_task = tasks[0]
    email_task = tasks[1]

    assert research_task.id == 1
    assert research_task.task_type == "retrieve"
    assert research_task.depends_on == []

    assert email_task.id == 2
    assert email_task.task_type == "email"
    assert email_task.depends_on == [1]