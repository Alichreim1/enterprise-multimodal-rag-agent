from config.bootstrap import build_rag_pipeline
from rag.agent.final_answer import FinalAnswerGenerator
from rag.agent.rag_agent import RAGAgent
from rag.agent.task_decomposer import TaskDecomposer
from rag.agents.agent_registry import AgentRegistry
from rag.agents.email_agent import EmailAgent
from rag.agents.research_agent import ResearchAgent
from rag.agents.supervisor import Supervisor
from rag.llms.ollama_llm import OllamaLLM
from rag.llms.openai_llm import OpenAILLM


class ContextAwareRAGLLM:

    def generate(self, prompt: str) -> str:

        assert "ali@example.com" in prompt

        return (
            "Ali's email address is "
            "ali@example.com."
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


class FakeMemoryManager:

    def recall_relevant(
        self,
        query: str,
        k: int,
    ):
        return []


class FailingLegacyComponent:

    def __getattr__(self, name):
        raise AssertionError(
            "Legacy components must not handle "
            "retrieve/email tasks."
        )


def test_real_multi_agent_stream(
    tmp_path,
):

    document_path = (
        tmp_path
        / "ali_profile.txt"
    )

    document_path.write_text(
        (
            "Ali Chreim is an AI engineer.\n"
            "Ali's email address is "
            "ali@example.com.\n"
            "Ali builds RAG systems."
        ),
        encoding="utf-8",
    )

    # Real retrieval infrastructure
    rag_pipeline = build_rag_pipeline(
        file_path=document_path,
        answer_llm=ContextAwareRAGLLM(),
    )

    # Real reasoning model
    reasoning_llm = OpenAILLM(
        model="gpt-5.6-luna",
    )

    task_decomposer = TaskDecomposer(
        llm=reasoning_llm,
    )

    # Specialized agents
    email_tool = FakeEmailTool()

    research_agent = ResearchAgent(
        rag=rag_pipeline,
    )

    email_agent = EmailAgent(
        email_tool=email_tool,
        llm=reasoning_llm,
    )

    registry = AgentRegistry()

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

    # Real final-answer model
    final_answer_generator = (
        FinalAnswerGenerator(
            llm=OllamaLLM(
                model="qwen3:8B",
            ),
        )
    )

    legacy_component = (
        FailingLegacyComponent()
    )

    agent = RAGAgent(
        task_decomposer=task_decomposer,
        planner=legacy_component,
        executor=legacy_component,
        reflection=legacy_component,
        final_answer_generator=(
            final_answer_generator
        ),
        memory_manager=FakeMemoryManager(),
        supervisor=supervisor,
    )

    question = (
        "Find Ali's email and send him "
        'an email with subject "Meeting" '
        'and body "The meeting is tomorrow '
        'at 5 PM."'
    )

    events = list(
        agent.stream(question)
    )

    event_types = [
        event["type"]
        for event in events
    ]

    print("\n=== STREAM EVENTS ===")

    for event in events:
        print(
            event["type"],
            "|",
            event.get("message"),
            "|",
            event.get("result"),
        )

    assert "memory_recall_started" in event_types
    assert "memory_recall_completed" in event_types

    assert (
        "task_decomposition_started"
        in event_types
    )

    assert "tasks_created" in event_types

    assert event_types.count(
        "task_started"
    ) == 2

    assert event_types.count(
        "execution_started"
    ) == 2

    assert event_types.count(
        "execution_completed"
    ) == 2

    assert event_types.count(
        "task_completed"
    ) == 2

    assert "planning_started" not in event_types
    assert "reflection_started" not in event_types
    assert "agent_error" not in event_types

    assert "final_answer_started" in event_types
    assert "final_answer_chunk" in event_types
    assert "final_answer" in event_types

    assert event_types[-1] == "agent_finished"

    execution_events = [
        event
        for event in events
        if (
            event["type"]
            == "execution_completed"
        )
    ]

    assert (
        execution_events[0]
        ["metadata"]["mode"]
        == "multi_agent"
    )

    assert (
        execution_events[0]
        ["metadata"]["data"]["email"]
        == "ali@example.com"
    )

    assert (
        execution_events[1]
        ["metadata"]["data"]["recipient"]
        == "ali@example.com"
    )

    assert email_tool.called_with == {
        "to": "ali@example.com",
        "subject": "Meeting",
        "body": (
            "The meeting is tomorrow at 5 PM."
        ),
    }

    final_answer_event = next(
        event
        for event in events
        if event["type"] == "final_answer"
    )

    final_answer = (
        final_answer_event["result"]
    )

    assert isinstance(final_answer, str)
    assert final_answer.strip()

    assert (
        "ali@example.com"
        in final_answer.lower()
    )