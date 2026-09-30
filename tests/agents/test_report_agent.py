from rag.agents.report_agent import (
    ReportAgent,
)
from rag.models.context import TaskContext
from rag.models.result import AgentResult


class FakeLLM:

    def __init__(self):
        self.received_prompt = None

    def generate(self, prompt: str):
        self.received_prompt = prompt

        return """
        {
          "title": "Python Update",
          "content": "Python 3.14.7 is the latest stable Python version and was released on August 5, 2026."
        }
        """


def test_report_agent_generates_report():

    llm = FakeLLM()

    agent = ReportAgent(
        llm=llm,
    )

    web_result = AgentResult(
        status="success",
        data={
            "query": (
                "Find the latest stable "
                "Python version."
            ),
            "answer": (
                "Python 3.14.7 is the latest "
                "stable Python version and was "
                "released on August 5, 2026."
            ),
        },
    )

    context = TaskContext(
        task_id=2,
        task_description=(
            "Write a short report about the "
            "latest stable Python version."
        ),
        dependency_results={
            1: web_result,
        },
    )

    result = agent.run(context)

    assert result.status == "success"

    assert result.data == {
        "title": "Python Update",
        "content": (
            "Python 3.14.7 is the latest "
            "stable Python version and was "
            "released on August 5, 2026."
        ),
    }

    assert result.message == (
        "Report generated successfully."
    )

    assert "Python 3.14.7" in (
        llm.received_prompt
    )


def test_report_agent_requires_dependencies():

    agent = ReportAgent(
        llm=FakeLLM(),
    )

    context = TaskContext(
        task_id=1,
        task_description=(
            "Write a Python report."
        ),
        dependency_results={},
    )

    result = agent.run(context)

    assert result.status == "failure"

    assert result.message == (
        "No dependency results are "
        "available for the report."
    )


def test_report_agent_rejects_failed_dependency():

    agent = ReportAgent(
        llm=FakeLLM(),
    )

    failed_result = AgentResult(
        status="failure",
        message="Web search failed.",
    )

    context = TaskContext(
        task_id=2,
        task_description=(
            "Write a Python report."
        ),
        dependency_results={
            1: failed_result,
        },
    )

    result = agent.run(context)

    assert result.status == "failure"

    assert result.data[
        "failed_dependencies"
    ] == [1]


def test_report_agent_handles_invalid_json():

    class InvalidLLM:

        def generate(self, prompt: str):
            return "This is not JSON."

    agent = ReportAgent(
        llm=InvalidLLM(),
    )

    context = TaskContext(
        task_id=2,
        task_description=(
            "Write a Python report."
        ),
        dependency_results={
            1: AgentResult(
                status="success",
                data={
                    "answer": "Python information.",
                },
            ),
        },
    )

    result = agent.run(context)

    assert result.status == "failure"

    assert result.message == (
        "Could not parse the generated report."
    )