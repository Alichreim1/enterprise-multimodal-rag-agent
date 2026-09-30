from datetime import datetime

from rag.memory.memory import SimpleMemory
from rag.memory.models import MemoryItem


def test_memory_add_and_retrieve():

    memory = SimpleMemory()

    item = MemoryItem(
        id="1",
        content="User prefers Python",
        memory_type="preference",
        created_at=datetime.now(),
    )

    memory.add(item)

    results = memory.retrieve(
        "What programming language does the user prefer?"
    )

    assert len(results) == 1
    assert results[0].content == "User prefers Python"


def test_memory_delete():

    memory = SimpleMemory()

    item = MemoryItem(
        id="1",
        content="User prefers Python",
        memory_type="preference",
        created_at=datetime.now(),
    )

    memory.add(item)

    memory.delete("1")

    results = memory.retrieve("Python")

    assert len(results) == 0


def test_memory_clear():

    memory = SimpleMemory()

    memory.add(
        MemoryItem(
            id="1",
            content="User prefers Python",
            memory_type="preference",
            created_at=datetime.now(),
        )
    )

    memory.clear()

    assert len(memory.memories) == 0