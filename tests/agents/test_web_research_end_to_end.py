import re

import pytest

from config.bootstrap import build_rag_pipeline
from rag.agent.final_answer import (
    FinalAnswerGenerator,
)
from rag.agent.state import AgentState
from rag.agent.task_decomposer import (
    TaskDecomposer,
)
from rag.agents.agent_registry import AgentRegistry
from rag.agents.email_agent import EmailAgent
from rag.agents.research_agent import ResearchAgent
from rag.agents.supervisor import Supervisor
from rag.agents.web_research_agent import (
    WebResearchAgent,
)
from rag.llms.ollama_llm import OllamaLLM
from rag.llms.openai_llm import OpenAILLM
from rag.tools.web_search_tool import (
    WebSearchTool,
)


class FakeEmailTool:

    def __init__(self):
        self.called_with = None

    def run(
        self,
        to: str,
        subject: str,
        body: str,
    ):
        self.called_with = {
            "to": to,
            "subject": subject,
            "body": body,
        }

        return "Email sent successfully."


@pytest.mark.integration
def test_web_research_end_to_end(
    tmp_path,
):

    # -----------------------------------------
    # Real test document
    # -----------------------------------------

    document_path = (
        tmp_path
        / "ali_profile.txt"
    )

    document_path.write_text(
        (
            "Ali Chreim is an AI engineer.\n"
            "Ali's email address is "
            "ali@example.com.\n"
        ),
        encoding="utf-8",
    )

    # -----------------------------------------
    # Real models and tools
    # -----------------------------------------

    reasoning_llm = OpenAILLM(
        model="gpt-5.6-luna",
    )

    rag_pipeline = build_rag_pipeline(
        file_path=document_path,
        answer_llm=OllamaLLM(
            model="qwen2.5:3b",
        ),
    )

    web_search_tool = WebSearchTool(
        model="gpt-5.6-luna",
    )

    email_tool = FakeEmailTool()

    # -----------------------------------------
    # Real TaskDecomposer
    # -----------------------------------------

    task_decomposer = TaskDecomposer(
        llm=reasoning_llm,
    )

    question = (
        "Find the latest stable Python "
        "version from the web, find Ali's "
        "email in the uploaded document, "
        "and send him the result with "
        'subject "Python Update".'
    )

    tasks = task_decomposer.decompose(
        question
    )

    # -----------------------------------------
    # Real specialized agents
    # -----------------------------------------

    web_research_agent = (
        WebResearchAgent(
            web_search_tool=(
                web_search_tool
            ),
        )
    )

    research_agent = ResearchAgent(
        rag=rag_pipeline,
    )

    email_agent = EmailAgent(
        email_tool=email_tool,
        llm=reasoning_llm,
    )

    # -----------------------------------------
    # Registry and Supervisor
    # -----------------------------------------

    registry = AgentRegistry()

    registry.register(
        task_type="web_search",
        agent=web_research_agent,
    )

    registry.register(
        task_type="retrieve",
        agent=research_agent,
    )

    registry.register(
        task_type="email",
        agent=email_agent,
    )

    supervisor = Supervisor(
        registry=registry,
    )

    state = AgentState(
        question=question,
        tasks=tasks,
    )

    # -----------------------------------------
    # Execute all tasks
    # -----------------------------------------

    supervisor.run(state)

    print("\n=== TASK RESULTS ===")

    for task in state.tasks:
        print(
            f"Task {task.id} | "
            f"type={task.task_type} | "
            f"result={task.result}"
        )

    # -----------------------------------------
    # Find tasks without assuming order
    # -----------------------------------------

    web_task = next(
        task
        for task in state.tasks
        if task.task_type == "web_search"
    )

    retrieve_task = next(
        task
        for task in state.tasks
        if task.task_type == "retrieve"
    )

    email_task = next(
        task
        for task in state.tasks
        if task.task_type == "email"
    )

    # -----------------------------------------
    # Validate web research
    # -----------------------------------------

    web_result = state.results[
        web_task.id
    ]

    assert web_result.status == "success"

    web_answer = web_result.data["answer"]

    assert "python" in web_answer.lower()

    version_match = re.search(
        r"Python\s+\d+(?:\.\d+)+",
        web_answer,
        flags=re.IGNORECASE,
    )

    assert version_match is not None

    discovered_version = (
        version_match.group(0)
    )

    # -----------------------------------------
    # Validate RAG recipient
    # -----------------------------------------

    retrieve_result = state.results[
        retrieve_task.id
    ]

    assert retrieve_result.status == "success"

    assert (
        retrieve_result.data["email"]
        == "ali@example.com"
    )

    # -----------------------------------------
    # Validate dependencies
    # -----------------------------------------

    assert set(
        email_task.depends_on
    ) == {
        web_task.id,
        retrieve_task.id,
    }

    # -----------------------------------------
    # Validate email execution
    # -----------------------------------------

    assert email_task.completed is True

    assert email_tool.called_with is not None

    assert email_tool.called_with["to"] == (
        "ali@example.com"
    )

    assert email_tool.called_with["subject"] == (
        "Python Update"
    )

    assert (
        discovered_version.lower()
        in email_tool.called_with[
            "body"
        ].lower()
    )

    # -----------------------------------------
    # Real final answer
    # -----------------------------------------

    final_answer_generator = (
        FinalAnswerGenerator(
            llm=OllamaLLM(
                model="qwen3:8B",
            ),
        )
    )

    final_answer = (
        final_answer_generator.generate(
            question=question,
            tasks=state.tasks,
            memories=[],
        )
    )

    print(
        "\n=== FINAL ANSWER ===\n"
        f"{final_answer}"
    )

    normalized_answer = (
        final_answer.lower()
    )

    assert final_answer.strip()

    assert (
        "ali@example.com"
        in normalized_answer
    )

    assert (
        discovered_version.lower()
        in normalized_answer
    )

    assert (
        "sent" in normalized_answer
        or "success" in normalized_answer
    )