import logging
import uuid
from pathlib import Path

from rag.chunkers.image_chunker import ImageChunker
from rag.chunkers.recursive_chunker import RecursiveChunker
from rag.loaders.loader_factory import LoaderFactory
from rag.pipeline.multimodal_rag_pipeline import MultimodalRAGPipeline
from rag.retrievers.multimodal_retriever import MultimodalRetriever
from rag.retrievers.similarity_retriever import SimilarityRetriever
from rag.vectorstores.faiss_vector_store import FAISSVectorStore


logger = logging.getLogger(__name__)


class KnowledgeBaseManager:
    """
    Manages one shared multimodal knowledge base.

    Text chunks and image chunks are stored separately
    because they use different embedding models.
    """

    def __init__(
        self,
        text_embedder,
        image_embedder,
        text_chunker=None,
        image_chunker=None,
        text_top_k: int = 2,
        image_top_k: int = 3,
        min_score: float = 0.0,
    ):
        # ==================================================
        # Embedders
        # ==================================================

        self.text_embedder = text_embedder
        self.image_embedder = image_embedder

        # ==================================================
        # Chunkers
        # ==================================================

        self.text_chunker = text_chunker or RecursiveChunker()
        self.image_chunker = image_chunker or ImageChunker()

        # ==================================================
        # Separate Vector Stores
        # ==================================================

        self.text_vector_store = FAISSVectorStore(
            dimension=text_embedder.dimension,
        )

        self.image_vector_store = FAISSVectorStore(
            dimension=image_embedder.dimension,
        )

        # ==================================================
        # Separate Retrievers
        # ==================================================

        self.text_retriever = SimilarityRetriever(
            embedder=text_embedder,
            vector_store=self.text_vector_store,
            top_k=text_top_k,
            min_score=min_score,
        )

        self.image_retriever = SimilarityRetriever(
            embedder=image_embedder,
            vector_store=self.image_vector_store,
            top_k=image_top_k,
            min_score=min_score,
        )

        # ==================================================
        # Combined Multimodal Retriever
        # ==================================================

        self.retriever = MultimodalRetriever(
            text_retriever=self.text_retriever,
            image_retriever=self.image_retriever,
        )

        # ==================================================
        # Uploaded Document Registry
        # ==================================================

        self._documents: dict[str, dict] = {}

    # ======================================================
    # Add One File
    # ======================================================

    def add_file(
        self,
        file_path: str | Path,
        original_filename: str | None = None,
    ) -> dict:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File does not exist: {file_path}",
            )

        document_id = uuid.uuid4().hex

        visible_filename = original_filename or file_path.name

        logger.info(
            "Adding file to knowledge base: %s",
            visible_filename,
        )

        # ==================================================
        # Load
        # ==================================================

        loader = LoaderFactory.get_loader(file_path)
        documents = loader.load(file_path)

        # ==================================================
        # Add Shared Metadata
        # ==================================================

        for document in documents:
            document.metadata.update(
                {
                    "document_id": document_id,
                    "source": visible_filename,
                    "original_filename": visible_filename,
                    "stored_path": str(file_path),
                }
            )

        # ==================================================
        # Separate Modalities
        # ==================================================

        text_documents = [
            document
            for document in documents
            if document.is_text
        ]

        image_documents = [
            document
            for document in documents
            if document.is_image
        ]

        # ==================================================
        # Chunk
        # ==================================================

        text_chunks = self.text_chunker.chunk(text_documents)
        image_chunks = self.image_chunker.chunk(image_documents)

        # ==================================================
        # Embed Before Mutating Stores
        # ==================================================

        embedded_text_chunks = []

        if text_chunks:
            embedded_text_chunks = (
                self.text_embedder.embed_documents(text_chunks)
            )

        embedded_image_chunks = []

        if image_chunks:
            embedded_image_chunks = (
                self.image_embedder.embed_documents(image_chunks)
            )

        # ==================================================
        # Append To Existing Stores
        # ==================================================

        if embedded_text_chunks:
            self.text_vector_store.add(embedded_text_chunks)

        if embedded_image_chunks:
            self.image_vector_store.add(embedded_image_chunks)

        # ==================================================
        # Store File Information
        # ==================================================

        file_record = {
            "document_id": document_id,
            "filename": visible_filename,
            "stored_path": str(file_path),
            "text_chunks": len(text_chunks),
            "image_chunks": len(image_chunks),
        }

        self._documents[document_id] = file_record

        logger.info(
            "File added successfully. "
            "filename=%s text_chunks=%s image_chunks=%s",
            visible_filename,
            len(text_chunks),
            len(image_chunks),
        )

        return file_record

    # ======================================================
    # Retrieve
    # ======================================================

    def retrieve(self, query: str):
        return self.retriever.retrieve(query)

    # ======================================================
    # Build Shared Pipeline
    # ======================================================

    def build_pipeline(
        self,
        prompt_builder,
        llm,
        text_llm=None,
        text_prompt_builder=None,
    ) -> MultimodalRAGPipeline:
        return MultimodalRAGPipeline(
            retriever=self.retriever,
            prompt_builder=prompt_builder,
            llm=llm,
            text_llm=text_llm,
            text_prompt_builder=text_prompt_builder,
        )

    # ======================================================
    # Statistics
    # ======================================================

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def text_chunk_count(self) -> int:
        return self.text_vector_store.size

    @property
    def image_chunk_count(self) -> int:
        return self.image_vector_store.size

    @property
    def documents(self) -> list[dict]:
        return list(self._documents.values())
