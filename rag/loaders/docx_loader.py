from pathlib import Path

from docx import Document as WordDocument

from rag.models.document import Document
from rag.loaders.base_loader import BaseLoader


class DOCXLoader(BaseLoader):

    def load(self, file_path: Path) -> list[Document]:

        doc = WordDocument(file_path)

        text = "\n".join(
            paragraph.text
            for paragraph in doc.paragraphs
            if paragraph.text.strip()
        )

        return [
            Document(
                content=text,
                metadata={
                    "source": file_path.name,
                    "page": 1,
                    "file_type": "docx",
                },
            )
        ]