from pathlib import Path

from rag.loaders.pdf_loader import PDFLoader
from rag.loaders.docx_loader import DOCXLoader
from rag.loaders.txt_loader import TXTLoader
from rag.loaders.base_loader import BaseLoader
from rag.loaders.image_loader import (
    ImageLoader,
)

class LoaderFactory:

    LOADERS = {
        ".pdf": PDFLoader,
        ".docx": DOCXLoader,
        ".txt": TXTLoader,
        ".png": ImageLoader,
        ".jpg": ImageLoader,
        ".jpeg": ImageLoader,
        ".webp": ImageLoader,

    }

    @classmethod
    def get_loader(cls, file_path: Path) -> BaseLoader:

        extension = file_path.suffix.lower()

        if extension not in cls.LOADERS:
            raise ValueError(
                f"Unsupported file type: {extension}"
            )

        return cls.LOADERS[extension]()