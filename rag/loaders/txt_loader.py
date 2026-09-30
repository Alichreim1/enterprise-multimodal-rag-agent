from pathlib import Path

from rag.models.document import Document
from rag.loaders.base_loader import BaseLoader


class TXTLoader(BaseLoader):

    def load(self, file_path: Path) -> list[Document]:

        text = file_path.read_text(
            encoding="utf-8"
        )

        return [
            Document(
                content=text,
                metadata={
                    "source": file_path.name,
                    "page": 1,
                    "file_type": "txt",
                },
            )
        ]