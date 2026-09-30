from pathlib import Path

from PIL import Image
from PIL import UnidentifiedImageError

from rag.loaders.base_loader import BaseLoader
from rag.models.document import Document


class ImageLoader(BaseLoader):

    MIME_TYPES = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }

    def load(
        self,
        file_path: Path,
    ) -> list[Document]:

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Image not found: {file_path}"
            )

        extension = (
            file_path.suffix.lower()
        )

        if extension not in self.MIME_TYPES:
            raise ValueError(
                "Unsupported image type: "
                f"{extension}"
            )

        try:
            with Image.open(
                file_path
            ) as image:

                image.verify()

            with Image.open(
                file_path
            ) as image:

                width, height = image.size
                image_format = image.format

        except UnidentifiedImageError as exc:
            raise ValueError(
                f"Invalid image file: "
                f"{file_path}"
            ) from exc

        document = Document(
            content="",
            content_type="image",
            image_path=str(file_path),
            mime_type=(
                self.MIME_TYPES[extension]
            ),
            metadata={
                "source": file_path.name,
                "file_path": str(file_path),
                "extension": extension,
                "image_format": image_format,
                "width": width,
                "height": height,
            },
        )

        return [document]