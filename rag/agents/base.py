from abc import ABC, abstractmethod

from rag.models.context import TaskContext
from rag.models.result import AgentResult


class BaseAgent(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def run(self, context: TaskContext) -> AgentResult:
        pass