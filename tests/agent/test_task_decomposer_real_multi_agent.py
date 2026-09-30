from rag.agent.task_decomposer import (
    TaskDecomposer,
)
from rag.llms.openai_llm import OpenAILLM


def test_real_decomposer_creates_email_dependency():

    llm = OpenAILLM(
        model="gpt-5.6-luna",
    )

    decomposer = TaskDecomposer(
        llm=llm,
    )

    question = (
        "Find Ali's email and send him an email "
        'with subject "Meeting" and body '
        '"The meeting is tomorrow at 5 PM."'
    )

    tasks = decomposer.decompose(
        question
    )

    print("\n=== REAL DECOMPOSED TASKS ===")

    for task in tasks:
        print(
            f"id={task.id} | "
            f"type={task.task_type} | "
            f"depends_on={task.depends_on} | "
            f"description={task.description}"
        )

    assert len(tasks) == 2

    research_task = tasks[0]
    email_task = tasks[1]

    assert research_task.task_type == "retrieve"
    assert research_task.depends_on == []

    assert email_task.task_type == "email"

    assert email_task.depends_on == [
        research_task.id
    ]

    email_description = (
        email_task.description.lower()
    )

    assert "meeting" in email_description
    assert "tomorrow" in email_description
    assert "5 pm" in email_description