from pathlib import Path

from rag.agents.research_agent import ResearchAgent
from rag.models.context import TaskContext
from rag.pipeline.rag_pipeline import RAGPipeline

from config.bootstrap import build_rag_pipeline


def test_research_agent_with_real_rag(tmp_path):

    # -----------------------------------------
    # Create a small test document
    # -----------------------------------------

    document = tmp_path / "test_document.txt"

    document.write_text(
        """
        Ali Hassan Chreim is a Machine Learning Engineer.

        Ali's email is ali@example.com.

        Ali's favorite programming language is Python.
        """,
        encoding="utf-8",
    )

    # -----------------------------------------
    # Build the real RAG pipeline
    # -----------------------------------------

    rag = build_rag_pipeline(
        file_path=document,
    )

    assert isinstance(rag, RAGPipeline)

    # -----------------------------------------
    # Create ResearchAgent
    # -----------------------------------------

    agent = ResearchAgent(rag)

    # -----------------------------------------
    # Create task context
    # -----------------------------------------

    context = TaskContext(
        task_id=1,
        task_description="What is Ali's email?",
    )

    # -----------------------------------------
    # Execute ResearchAgent
    # -----------------------------------------

    result = agent.run(context)

    # -----------------------------------------
    # Validate AgentResult
    # -----------------------------------------

    assert result.status == "success"
    assert "answer" in result.data

    answer = result.data["answer"]

    assert "ali@example.com" in answer.lower()