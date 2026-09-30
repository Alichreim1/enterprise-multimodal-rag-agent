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


DOCUMENT_PATH = "data/documents/AliChreimcv.pdf"


def build_rag():

    file_path = Path(DOCUMENT_PATH)

    # =========================
    # Load
    # =========================

    loader = LoaderFactory.get_loader(file_path)
    documents = loader.load(file_path)

    # =========================
    # Chunk
    # =========================

    chunker = RecursiveChunker(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = chunker.chunk(documents)

    # =========================
    # Embeddings
    # =========================

    embedder = SentenceTransformerEmbedder(
        EMBEDDING_MODEL
    )

    # =========================
    # Vector Store
    # =========================

    vector_store = FAISSVectorStore(
        dimension=embedder.dimension
    )

    embedded_chunks = embedder.embed_documents(
        chunks
    )

    vector_store.add(embedded_chunks)

    # =========================
    # Retriever
    # =========================

    retriever = SimilarityRetriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    # =========================
    # Prompt
    # =========================

    prompt_loader = PromptLoader()

    prompt_builder = RAGPromptBuilder(
        prompt_loader=prompt_loader
    )

    # =========================
    # LLM
    # =========================

    llm = OllamaLLM(
        model=LLM_MODEL
    )

    # =========================
    # RAG
    # =========================

    rag = RAGPipeline(
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
    )

    return rag


def main():

    rag = build_rag()

    questions = [
        "What is Ali's GitHub?",
        "What is Ali's LinkedIn?",
        "What is Ali's favorite programming language?",
        "What is Ali's favorite movie?",
    ]

    for question in questions:

        print("\n")
        print("=" * 80)
        print("QUESTION")
        print("=" * 80)
        print(question)

        # =========================
        # Retrieval
        # =========================

        results = rag.retrieve_context(question)

        print("\n")
        print("=" * 80)
        print("RETRIEVED RESULTS")
        print("=" * 80)

        for i, result in enumerate(results, 1):

            print(f"\n--- RESULT {i} ---")
            print(result)

        # =========================
        # Full RAG
        # =========================

        response = rag.ask(question)

        print("\n")
        print("=" * 80)
        print("RAG ANSWER")
        print("=" * 80)
        print(response.answer)


if __name__ == "__main__":
    main()