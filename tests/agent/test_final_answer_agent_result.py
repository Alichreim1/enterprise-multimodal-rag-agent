from rag.agent.final_answer import (
    FinalAnswerGenerator,
)
from rag.models.result import AgentResult
from rag.models.task import AgentTask


class FakeLLM:

    def __init__(self):
        self.received_prompt = None

    def generate(self, prompt: str) -> str:
        self.received_prompt = prompt

        return (
            "The email was sent successfully "
            "to ali@example.com."
        )


def test_final_answer_uses_agent_result():

    llm = FakeLLM()

    generator = FinalAnswerGenerator(
        llm=llm,
    )

    research_task = AgentTask(
        id=1,
        description="Find Ali's email.",
        task_type="retrieve",
        completed=True,
        result=AgentResult(
            status="success",
            data={
                "answer": (
                    "Ali's email is "
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
        description="Send Ali the meeting email.",
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

    assert answer == (
        "The email was sent successfully "
        "to ali@example.com."
    )

    prompt = llm.received_prompt

    assert '"email": "ali@example.com"' in prompt
    assert '"recipient": "ali@example.com"' in prompt
    assert '"subject": "Meeting"' in prompt
    assert '"tool_result": "Email sent successfully."' in prompt

    # Ensure Python dataclass representation
    # was not sent to the final LLM.
    assert "AgentResult(" not in prompt

def test_final_answer_supports_legacy_string_result():

    llm = FakeLLM()

    generator = FinalAnswerGenerator(
        llm=llm,
    )

    task = AgentTask(
        id=1,
        description="Calculate 2 + 2.",
        task_type="tool",
        completed=True,
        result="4",
    )

    generator.generate(
        question="What is 2 + 2?",
        tasks=[task],
        memories=[],
    )

    assert "Result:\n4" in llm.received_prompt    