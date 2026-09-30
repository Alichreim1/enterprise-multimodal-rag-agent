from pathlib import Path
import logging

from pypdf import PdfReader

from rag.loaders.base_loader import BaseLoader
from rag.models.document import Document


logger = logging.getLogger(__name__)


class PDFLoader(BaseLoader):
    def load(self, file_path: Path) -> list[Document]:
        logger.info(
            "Loading PDF: %s",
            file_path.name,
        )

        reader = PdfReader(file_path)
        documents = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if not text or not text.strip():
                continue

            document = Document(
                content=text,
                metadata={
                    "source": file_path.name,
                    "page": page_number,
                    "file_type": "pdf",
                },
            )

            documents.append(document)

        logger.info(
            "Loaded %d pages.",
            len(documents),
        )

        return documents