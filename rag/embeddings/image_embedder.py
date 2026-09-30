import logging
from pathlib import Path

from PIL import Image
from sentence_transformers import (
    SentenceTransformer,
)

from rag.embeddings.base_embedder import (
    BaseEmbedder,
)
from rag.models.chunk import Chunk
from rag.models.embedded_chunk import (
    EmbeddedChunk,
)


logger = logging.getLogger(__name__)


class ImageEmbedder(BaseEmbedder):

    def __init__(
        self,
        model_name: str = (
            "sentence-transformers/"
            "clip-ViT-B-32"
        ),
        model=None,
    ):
        # model injection keeps unit tests fast.
        self.model = (
            model
            if model is not None
            else SentenceTransformer(
                model_name
            )
        )

        self.dimension = (
            self.model.get_embedding_dimension()
        )

        if self.dimension is None:
            raise ValueError(
                "Could not determine image "
                "embedding dimension."
            )

    # ==================================================
    # Image Embeddings
    # ==================================================

    def embed_documents(
        self,
        chunks: list[Chunk],
    ) -> list[EmbeddedChunk]:

        image_chunks = [
            chunk
            for chunk in chunks
            if chunk.is_image
        ]

        if not image_chunks:
            return []

        logger.info(
            "Generating embeddings for "
            "%d image chunks...",
            len(image_chunks),
        )

        images = []

        for chunk in image_chunks:

            image_path = Path(
                chunk.image_path
            )

            if not image_path.exists():
                raise FileNotFoundError(
                    "Image chunk file "
                    f"not found: {image_path}"
                )

            with Image.open(
                image_path
            ) as image:

                # copy() makes the image independent
                # from the closed file handle.
                rgb_image = (
                    image
                    .convert("RGB")
                    .copy()
                )

                images.append(
                    rgb_image
                )

        vectors = self.model.encode(
            images,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        embedded_chunks = []

        for chunk, vector in zip(
            image_chunks,
            vectors,
        ):
            embedded_chunks.append(
                EmbeddedChunk(
                    chunk=chunk,
                    vector=vector,
                )
            )

        logger.info(
            "Image embedding generation "
            "completed."
        )

        return embedded_chunks

    # ==================================================
    # Text-To-Image Query Embedding
    # ==================================================

    def embed_query(
        self,
        query: str,
    ) -> list[float]:

        query = query.strip()

        if not query:
            raise ValueError(
                "Image search query "
                "cannot be empty."
            )

        vector = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return vector.tolist()