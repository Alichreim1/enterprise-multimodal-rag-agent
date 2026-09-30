from pathlib import Path

from config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    LLM_MODEL,
)

from rag.chunkers.recursive_chunker import RecursiveChunker
from rag.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from rag.llms.ollama_llm import OllamaLLM
from rag.loaders.loader_factory import LoaderFactory
from rag.pipeline.rag_pipeline import RAGPipeline
from rag.prompts.loaders.prompt_loader import PromptLoader
from rag.prompts.rag_prompt_builder import RAGPromptBuilder
from rag.retrievers.similarity_retriever import SimilarityRetriever
from rag.vectorstores.faiss_vector_store import FAISSVectorStore


PDF_PATH = Path(
    "data/documents/AliChreimcv.pdf"
)


def create_real_rag_pipeline():

    # -------------------------
    # Loader
    # -------------------------

    loader = LoaderFactory.get_loader(
        PDF_PATH
    )

    documents = loader.load(
        PDF_PATH
    )

    assert documents

    # -------------------------
    # Chunker
    # -------------------------

    chunker = RecursiveChunker(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = chunker.chunk(
        documents
    )

    assert chunks

    # -------------------------
    # Embedder
    # -------------------------

    embedder = SentenceTransformerEmbedder(
        EMBEDDING_MODEL,
    )

    # -------------------------
    # Vector Store
    # -------------------------

    vector_store = FAISSVectorStore(
        dimension=embedder.dimension,
    )

    embedded_chunks = (
        embedder.embed_documents(
            chunks
        )
    )

    vector_store.add(
        embedded_chunks
    )

    assert vector_store.size == len(
        chunks
    )

    # -------------------------
    # Retriever
    # -------------------------

    retriever = SimilarityRetriever(
        embedder=embedder,
        vector_store=vector_store,
        top_k=5,
        min_score=0.00,
    )

    # -------------------------
    # Prompt Builder
    # -------------------------

    prompt_loader = PromptLoader()

    prompt_builder = RAGPromptBuilder(
        prompt_loader=prompt_loader,
    )

    # -------------------------
    # LLM
    # -------------------------

    llm = OllamaLLM(
        model=LLM_MODEL,
    )

    # -------------------------
    # Pipeline
    # -------------------------

    return RAGPipeline(
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
    )


def test_real_rag_pipeline():

    rag = create_real_rag_pipeline()

    question = (
        "What is Ali's favorite football team?"
        
    )

    response = rag.ask(
        question
    )

    # -------------------------
    # RAGResponse structure
    # -------------------------

    assert response.answer
    assert response.prompt
    assert response.search_results

    # -------------------------
    # Retrieval happened
    # -------------------------

    assert len(
        response.search_results
    ) > 0

    # -------------------------
    # Prompt contains question
    # -------------------------

    assert question in response.prompt

    # -------------------------
    # Prompt contains retrieved context
    # -------------------------

    first_result = (
        response.search_results[0]
    )

    assert (
        first_result.chunk.content
        in response.prompt
    )

    # -------------------------
    # The answer should contain
    # information related to
    # the retrieved context.
    # -------------------------

    print()
    print("=" * 70)
    print("QUESTION:")
    print(question)

    print()
    print("TOP RETRIEVED RESULT:")
    print(
        first_result.chunk.content
    )

    print()
    print("SCORE:")
    print(
        first_result.score
    )

    print()
    print("ANSWER:")
    print(
        response.answer
    )

    print("=" * 70)