from rag.agent.final_answer import (
    FinalAnswerGenerator,
)
from rag.llms.ollama_llm import OllamaLLM
from rag.models.result import AgentResult
from rag.models.task import AgentTask


def test_real_final_answer_with_agent_results():

    research_task = AgentTask(
        id=1,
        description="Find Ali's email.",
        task_type="retrieve",
        completed=True,
        result=AgentResult(
            status="success",
            data={
                "answer": (
                    "Ali's email address is "
                    "ali@example.com."
                ),
                "email": "ali@example.com",
            },
            message=(
                "Research completed successfully."
            ),
        ),
    )

    email_task = AgentTask(
        id=2,
        description=(
            "Send Ali an email with subject "
            "Meeting."
        ),
        task_type="email",
        completed=True,
        result=AgentResult(
            status="success",
            data={
                "recipient": "ali@example.com",
                "subject": "Meeting",
                "body": (
                    "The meeting is tomorrow "
                    "at 5 PM."
                ),
                "tool_result": (
                    "Email sent successfully."
                ),
            },
            message=(
                "Email sent successfully."
            ),
        ),
        depends_on=[1],
    )

    generator = FinalAnswerGenerator(
        llm=OllamaLLM(
            model="qwen3:8B",
        ),
    )

    answer = generator.generate(
        question=(
            "Find Ali's email and send him "
            "the meeting email."
        ),
        tasks=[
            research_task,
            email_task,
        ],
        memories=[],
    )

    print(
        "\n=== FINAL ANSWER ===\n"
        f"{answer}"
    )

    assert isinstance(answer, str)
    assert answer.strip()

    normalized_answer = answer.lower()

    assert "ali@example.com" in normalized_answer

    assert (
        "sent" in normalized_answer
        or "success" in normalized_answer
    )

    assert "agentresult" not in normalized_answer
    assert "supervisor" not in normalized_answer