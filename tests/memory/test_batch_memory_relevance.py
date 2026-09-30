from pathlib import Path
from types import SimpleNamespace

from rag.memory.memory_manager import (
    MemoryManager,
)


class FakeVectorMemory:

    def __init__(self, memories):
        self.memories = memories

    def retrieve(
        self,
        query,
        k,
    ):
        return self.memories[:k]


class FakeReasoningLLM:

    def __init__(self, response):
        self.response = response
        self.call_count = 0
        self.received_prompt = None

    def generate(self, prompt):
        self.call_count += 1
        self.received_prompt = prompt

        return self.response


def build_memory(
    content,
    memory_type,
):
    return SimpleNamespace(
        content=content,
        memory_type=memory_type,
    )


def test_recall_relevant_uses_one_llm_call():
    memories = [
        build_memory(
            "User likes basketball.",
            "preference",
        ),
        build_memory(
            (
                "User's favorite programming "
                "language is Python."
            ),
            "preference",
        ),
        build_memory(
            (
                "User is building a RAG "
                "agent in Python."
            ),
            "project",
        ),
    ]

    reasoning_llm = FakeReasoningLLM(
        response=(
            '{"relevant_ids": [2]}'
        )
    )

    manager = MemoryManager(
        extractor=None,
        vector_memory=FakeVectorMemory(
            memories
        ),
        index_path=Path(
            "unused.index"
        ),
        metadata_path=Path(
            "unused.pkl"
        ),
        reasoning_llm=reasoning_llm,
    )

    result = manager.recall_relevant(
        query=(
            "What is my favorite "
            "programming language?"
        ),
        k=3,
    )

    assert reasoning_llm.call_count == 1

    assert len(result) == 1

    assert result[0].content == (
        "User's favorite programming "
        "language is Python."
    )

    assert "CANDIDATE MEMORIES" in (
        reasoning_llm.received_prompt
    )


def test_batch_relevance_can_select_multiple():
    memories = [
        build_memory(
            "User likes basketball.",
            "preference",
        ),
        build_memory(
            (
                "User's favorite programming "
                "language is Python."
            ),
            "preference",
        ),
        build_memory(
            (
                "User is building a RAG "
                "agent in Python."
            ),
            "project",
        ),
    ]

    reasoning_llm = FakeReasoningLLM(
        response=(
            '{"relevant_ids": [2, 3]}'
        )
    )

    manager = MemoryManager(
        extractor=None,
        vector_memory=FakeVectorMemory(
            memories
        ),
        index_path=Path(
            "unused.index"
        ),
        metadata_path=Path(
            "unused.pkl"
        ),
        reasoning_llm=reasoning_llm,
    )

    result = manager.recall_relevant(
        query=(
            "What is my favorite language "
            "and what project am I building?"
        ),
        k=3,
    )

    assert reasoning_llm.call_count == 1

    assert [
        memory.content
        for memory in result
    ] == [
        (
            "User's favorite programming "
            "language is Python."
        ),
        (
            "User is building a RAG "
            "agent in Python."
        ),
    ]


def test_invalid_batch_response_is_safe():
    memories = [
        build_memory(
            "User likes basketball.",
            "preference",
        ),
    ]

    reasoning_llm = FakeReasoningLLM(
        response="invalid response"
    )

    manager = MemoryManager(
        extractor=None,
        vector_memory=FakeVectorMemory(
            memories
        ),
        index_path=Path(
            "unused.index"
        ),
        metadata_path=Path(
            "unused.pkl"
        ),
        reasoning_llm=reasoning_llm,
    )

    result = manager.recall_relevant(
        query="What is my favorite sport?",
        k=1,
    )

    assert result == []
    assert reasoning_llm.call_count == 1