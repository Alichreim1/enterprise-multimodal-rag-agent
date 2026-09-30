from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.chunkers.base_chunker import BaseChunker
from rag.models.chunk import Chunk
from rag.models.document import Document


class RecursiveChunker(BaseChunker):

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(
        self,
        documents: list[Document],
    ) -> list[Chunk]:

        chunks = []
        chunk_id = 0

        for document in documents:

            if not document.is_text:
                continue

            pieces = self.splitter.split_text(
                document.content
            )

            for piece in pieces:

                metadata = document.metadata.copy()

                metadata["chunk_id"] = chunk_id
                metadata["content_type"] = "text"

                chunks.append(
                    Chunk(
                        content=piece,
                        metadata=metadata,
                        content_type="text",
                        mime_type=document.mime_type,
                    )
                )

                chunk_id += 1

        return chunks
