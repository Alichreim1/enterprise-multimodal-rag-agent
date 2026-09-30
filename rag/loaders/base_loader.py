from abc import ABC, abstractmethod
from pathlib import Path

from rag.models.document import Document


class BaseLoader(ABC):

    @abstractmethod
    def load(self, file_path: Path) -> list[Document]:
        """
        Load a file and return a list of Documents.
        """
        pass