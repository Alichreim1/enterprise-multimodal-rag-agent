from rag.agent.task_decomposer import (
    TaskDecomposer,
)
from rag.llms.openai_llm import OpenAILLM


def test_real_decomposer_creates_web_flow():

    decomposer = TaskDecomposer(
        llm=OpenAILLM(
            model="gpt-5.6-luna",
        ),
    )

    question = (
        "Find the latest stable Python "
        "version from the web, find Ali's "
        "email in the uploaded document, "
        "and send him the result with "
        'subject "Python Update".'
    )

    tasks = decomposer.decompose(
        question
    )

    print("\n=== DECOMPOSED TASKS ===")

    for task in tasks:
        print(
            f"id={task.id} | "
            f"type={task.task_type} | "
            f"depends_on={task.depends_on} | "
            f"description={task.description}"
        )

    assert len(tasks) == 3

    task_types = {
        task.task_type
        for task in tasks
    }

    assert task_types == {
        "web_search",
        "retrieve",
        "email",
    }

    web_task = next(
        task
        for task in tasks
        if task.task_type == "web_search"
    )

    retrieve_task = next(
        task
        for task in tasks
        if task.task_type == "retrieve"
    )

    email_task = next(
        task
        for task in tasks
        if task.task_type == "email"
    )

    assert web_task.depends_on == []
    assert retrieve_task.depends_on == []

    assert set(
        email_task.depends_on
    ) == {
        web_task.id,
        retrieve_task.id,
    }

    email_description = (
        email_task.description.lower()
    )

    assert "python update" in email_description
    assert "email" in email_description