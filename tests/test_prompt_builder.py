from rag.models.chunk import Chunk
from rag.models.search_result import SearchResult
from rag.prompts.rag_prompt_builder import RAGPromptBuilder


class FakePromptLoader:

    def __init__(self, template: str):
        self.template = template
        self.last_template_name = None

    def load(self, template_name: str) -> str:
        self.last_template_name = template_name
        return self.template


def create_search_result(
    content: str,
    source: str = "test.pdf",
    page: int = 1,
    score: float = 0.9,
) -> SearchResult:

    chunk = Chunk(
        content=content,
        metadata={
            "source": source,
            "page": page,
            "file_type": "pdf",
            "chunk_id": 0,
        },
    )

    return SearchResult(
        chunk=chunk,
        score=score,
    )


def test_prompt_builder_loads_rag_prompt():

    template = (
        "Context:\n"
        "{context}\n\n"
        "Question:\n"
        "{question}"
    )

    prompt_loader = FakePromptLoader(
        template
    )

    builder = RAGPromptBuilder(
        prompt_loader=prompt_loader,
    )

    assert (
        prompt_loader.last_template_name
        == "rag_prompt"
    )


def test_prompt_builder_inserts_question():

    template = (
        "Context:\n"
        "{context}\n\n"
        "Question:\n"
        "{question}"
    )

    prompt_loader = FakePromptLoader(
        template
    )

    builder = RAGPromptBuilder(
        prompt_loader=prompt_loader,
    )

    question = "What is artificial intelligence?"

    prompt = builder.build(
        question=question,
        results=[],
    )

    assert question in prompt


def test_prompt_builder_includes_chunk_content():

    template = (
        "Context:\n"
        "{context}\n\n"
        "Question:\n"
        "{question}"
    )

    prompt_loader = FakePromptLoader(
        template
    )

    builder = RAGPromptBuilder(
        prompt_loader=prompt_loader,
    )

    result = create_search_result(
        content="Employees receive 20 days of annual leave."
    )

    prompt = builder.build(
        question="How many vacation days do employees receive?",
        results=[result],
    )

    assert (
        "Employees receive 20 days of annual leave."
        in prompt
    )


def test_prompt_builder_includes_source_and_page():

    template = (
        "Context:\n"
        "{context}\n\n"
        "Question:\n"
        "{question}"
    )

    prompt_loader = FakePromptLoader(
        template
    )

    builder = RAGPromptBuilder(
        prompt_loader=prompt_loader,
    )

    result = create_search_result(
        content="Employees receive 20 days of annual leave.",
        source="employee_handbook.pdf",
        page=7,
    )

    prompt = builder.build(
        question="How many vacation days?",
        results=[result],
    )

    assert "employee_handbook.pdf" in prompt
    assert "Page: 7" in prompt


def test_prompt_builder_handles_multiple_results():

    template = (
        "Context:\n"
        "{context}\n\n"
        "Question:\n"
        "{question}"
    )

    prompt_loader = FakePromptLoader(
        template
    )

    builder = RAGPromptBuilder(
        prompt_loader=prompt_loader,
    )

    results = [
        create_search_result(
            content="Employees receive 20 days of annual leave.",
            source="handbook.pdf",
            page=5,
        ),
        create_search_result(
            content="Leave requests must be submitted to HR.",
            source="handbook.pdf",
            page=6,
        ),
    ]

    prompt = builder.build(
        question="How do employees request leave?",
        results=results,
    )

    assert (
        "Employees receive 20 days of annual leave."
        in prompt
    )

    assert (
        "Leave requests must be submitted to HR."
        in prompt
    )

    assert "Page: 5" in prompt
    assert "Page: 6" in prompt

    assert "---------------------" in prompt


def test_prompt_builder_handles_empty_results():

    template = (
        "Context:\n"
        "{context}\n\n"
        "Question:\n"
        "{question}"
    )

    prompt_loader = FakePromptLoader(
        template
    )

    builder = RAGPromptBuilder(
        prompt_loader=prompt_loader,
    )

    prompt = builder.build(
        question="What is AI?",
        results=[],
    )

    assert (
        "No relevant context found."
        in prompt
    )


def test_prompt_builder_uses_custom_template_name():

    template = (
        "Context:\n"
        "{context}\n\n"
        "Question:\n"
        "{question}"
    )

    prompt_loader = FakePromptLoader(
        template
    )

    builder = RAGPromptBuilder(
        prompt_loader=prompt_loader,
        template_name="custom_prompt",
    )

    assert (
        prompt_loader.last_template_name
        == "custom_prompt"
    )