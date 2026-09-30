import pytest

from rag.agent.final_answer import FinalAnswerGenerator
from rag.llms.openai_llm import OpenAILLM
from rag.models.result import AgentResult
from rag.models.task import AgentTask


@pytest.mark.integration
def test_final_answer_with_openai():

    llm = OpenAILLM(
        model="gpt-5.6-luna",
    )

    generator = FinalAnswerGenerator(
        llm=llm,
    )

    task = AgentTask(
        id=1,
        description="Find Ali's email.",
        task_type="retrieve",
        completed=True,
        result=AgentResult(
            status="success",
            data={
                "email": "ali@example.com",
            },
            message=(
                "Research completed successfully."
            ),
        ),
    )

    answer = generator.generate(
        question="What is Ali's email?",
        tasks=[task],
        memories=[],
    )

    print()
    print("=== FINAL ANSWER ===")
    print(answer)

    assert isinstance(answer, str)
    assert answer.strip()
    assert "ali@example.com" in answer