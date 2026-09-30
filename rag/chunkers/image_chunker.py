from rag.chunkers.base_chunker import (
    BaseChunker,
)
from rag.models.chunk import Chunk
from rag.models.document import Document


class ImageChunker(BaseChunker):

    def chunk(
        self,
        documents: list[Document],
    ) -> list[Chunk]:

        chunks = []

        chunk_id = 0

        for document in documents:

            if not document.is_image:
                continue

            metadata = (
                document.metadata.copy()
            )

            metadata["chunk_id"] = chunk_id
            metadata["content_type"] = (
                "image"
            )

            chunks.append(
                Chunk(
                    content=document.content,
                    metadata=metadata,
                    content_type="image",
                    image_path=(
                        document.image_path
                    ),
                    mime_type=(
                        document.mime_type
                    ),
                )
            )

            chunk_id += 1

        return chunks