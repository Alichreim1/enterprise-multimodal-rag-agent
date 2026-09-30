import pytest

from rag.agent.task_decomposer import (
    TaskDecomposer,
)
from rag.llms.openai_llm import OpenAILLM


@pytest.mark.integration
def test_real_decomposer_creates_report_flow():

    decomposer = TaskDecomposer(
        llm=OpenAILLM(
            model="gpt-5.6-luna",
        ),
    )

    question = (
        "Find the latest stable Python "
        "version from the web, write a short "
        "report about it, find Ali's email "
        "in the uploaded document, and email "
        "the report to Ali with subject "
        '"Python Report".'
    )

    tasks = decomposer.decompose(
        question
    )

    print("\n=== REPORT FLOW TASKS ===")

    for task in tasks:
        print(
            f"id={task.id} | "
            f"type={task.task_type} | "
            f"depends_on={task.depends_on} | "
            f"description={task.description}"
        )

    assert len(tasks) == 4

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

    report_task = next(
        task
        for task in tasks
        if task.task_type == "report"
    )

    email_task = next(
        task
        for task in tasks
        if task.task_type == "email"
    )

    assert web_task.depends_on == []
    assert retrieve_task.depends_on == []

    assert report_task.depends_on == [
        web_task.id
    ]

    assert set(
        email_task.depends_on
    ) == {
        retrieve_task.id,
        report_task.id,
    }

    assert (
        "python report"
        in email_task.description.lower()
    )