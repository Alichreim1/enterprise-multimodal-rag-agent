from config.bootstrap import build_rag_pipeline
from rag.agent.rag_agent import RAGAgent
from rag.agent.task_decomposer import TaskDecomposer
from rag.agents.agent_registry import AgentRegistry
from rag.agents.email_agent import EmailAgent
from rag.agents.research_agent import ResearchAgent
from rag.agents.supervisor import Supervisor
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
            "Legacy components must not execute "
            "retrieve/email Multi-Agent tasks."
        )


class FakeFinalAnswerGenerator:

    def __init__(self):
        self.received_tasks = None

    def generate(
        self,
        question,
        tasks,
        memories,
    ):
        self.received_tasks = tasks

        email_result = tasks[1].result

        assert email_result.status == "success"

        return (
            "The email was sent successfully "
            "to ali@example.com."
        )


def test_multi_agent_end_to_end(
    tmp_path,
):

    # -----------------------------------------
    # Test document
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
            "Ali builds RAG systems."
        ),
        encoding="utf-8",
    )

    # -----------------------------------------
    # Real RAG infrastructure
    # -----------------------------------------

    rag_pipeline = build_rag_pipeline(
        file_path=document_path,
        answer_llm=ContextAwareRAGLLM(),
    )

    # -----------------------------------------
    # Real OpenAI reasoning LLM
    # -----------------------------------------

    reasoning_llm = OpenAILLM(
        model="gpt-5.6-luna",
    )

    # -----------------------------------------
    # Real TaskDecomposer
    # -----------------------------------------

    task_decomposer = TaskDecomposer(
        llm=reasoning_llm,
    )

    # -----------------------------------------
    # Real specialized agents
    # -----------------------------------------

    email_tool = FakeEmailTool()

    research_agent = ResearchAgent(
        rag=rag_pipeline,
    )

    email_agent = EmailAgent(
        email_tool=email_tool,
        llm=reasoning_llm,
    )

    # -----------------------------------------
    # Registry + Supervisor
    # -----------------------------------------

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

    # -----------------------------------------
    # Existing RAGAgent
    # -----------------------------------------

    legacy_component = (
        FailingLegacyComponent()
    )

    final_answer_generator = (
        FakeFinalAnswerGenerator()
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

    # -----------------------------------------
    # Execute user request
    # -----------------------------------------

    answer = agent.ask(
        (
            "Find Ali's email and send him "
            'an email with subject "Meeting" '
            'and body "The meeting is tomorrow '
            'at 5 PM."'
        )
    )

    # -----------------------------------------
    # Assertions
    # -----------------------------------------

    assert answer == (
        "The email was sent successfully "
        "to ali@example.com."
    )

    assert email_tool.called_with == {
        "to": "ali@example.com",
        "subject": "Meeting",
        "body": (
            "The meeting is tomorrow at 5 PM."
        ),
    }

    tasks = (
        final_answer_generator
        .received_tasks
    )

    assert len(tasks) == 2

    assert tasks[0].task_type == "retrieve"
    assert tasks[0].completed is True

    assert tasks[1].task_type == "email"
    assert tasks[1].depends_on == [
        tasks[0].id
    ]
    assert tasks[1].completed is True

    assert (
        tasks[0].result.data["email"]
        == "ali@example.com"
    )

    assert (
        tasks[1].result.data["recipient"]
        == "ali@example.com"
    )