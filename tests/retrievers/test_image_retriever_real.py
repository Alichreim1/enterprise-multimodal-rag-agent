import pytest
from PIL import Image

from rag.chunkers.image_chunker import (
    ImageChunker,
)
from rag.embeddings.image_embedder import (
    ImageEmbedder,
)
from rag.loaders.image_loader import (
    ImageLoader,
)
from rag.retrievers.similarity_retriever import (
    SimilarityRetriever,
)
from rag.vectorstores.faiss_vector_store import (
    FAISSVectorStore,
)


@pytest.mark.integration
def test_real_text_to_image_retrieval(
    tmp_path,
):

    # -----------------------------------------
    # Create Test Images
    # -----------------------------------------

    red_path = (
        tmp_path / "red_square.png"
    )

    blue_path = (
        tmp_path / "blue_square.png"
    )

    Image.new(
        mode="RGB",
        size=(224, 224),
        color=(255, 0, 0),
    ).save(red_path)

    Image.new(
        mode="RGB",
        size=(224, 224),
        color=(0, 0, 255),
    ).save(blue_path)

    # -----------------------------------------
    # Load Images
    # -----------------------------------------

    loader = ImageLoader()

    documents = []

    documents.extend(
        loader.load(red_path)
    )

    documents.extend(
        loader.load(blue_path)
    )

    assert len(documents) == 2

    # -----------------------------------------
    # Image Chunks
    # -----------------------------------------

    chunker = ImageChunker()

    chunks = chunker.chunk(
        documents
    )

    assert len(chunks) == 2

    # -----------------------------------------
    # Real CLIP Embeddings
    # -----------------------------------------

    embedder = ImageEmbedder(
        model_name=(
            "sentence-transformers/"
            "clip-ViT-B-32"
        )
    )

    embedded_chunks = (
        embedder.embed_documents(
            chunks
        )
    )

    assert len(embedded_chunks) == 2

    # -----------------------------------------
    # Separate Image FAISS Store
    # -----------------------------------------

    vector_store = FAISSVectorStore(
        dimension=embedder.dimension
    )

    vector_store.add(
        embedded_chunks
    )

    assert vector_store.size == 2

    # -----------------------------------------
    # Text-To-Image Retrieval
    # -----------------------------------------

    retriever = SimilarityRetriever(
        embedder=embedder,
        vector_store=vector_store,
        top_k=2,
        min_score=0.0,
    )

    results = retriever.retrieve(
        "A solid blue square image."
    )

    print(
        "\n=== IMAGE SEARCH RESULTS ==="
    )

    for result in results:
        print(
            result.chunk.image_path,
            result.score,
        )

    assert len(results) == 2

    best_result = results[0]

    assert best_result.chunk.is_image

    assert (
        best_result.chunk.metadata[
            "source"
        ]
        == "blue_square.png"
    )

    assert (
        best_result.chunk.image_path
        == str(blue_path)
    )

    assert (
        results[0].score
        > results[1].score
    )