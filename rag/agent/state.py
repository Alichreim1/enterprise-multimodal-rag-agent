
from dataclasses import dataclass, field

from rag.memory.models import MemoryItem
from rag.models.task import AgentTask
from rag.models.result import AgentResult


@dataclass
class AgentState:
    """
    Holds the mutable state of one agent execution.
    """

    # Original user question
    question: str

    # Tasks produced by the TaskDecomposer
    tasks: list[AgentTask] = field(default_factory=list)

    # Index of the task currently being processed
    current_task_index: int = 0

    # Number of agent execution iterations
    iteration: int = 0

    # Whether the complete agent execution has finished
    finished: bool = False

    memories: list[MemoryItem] = field(default_factory=list)

    results: dict[int, AgentResult] = field(default_factory=dict)

