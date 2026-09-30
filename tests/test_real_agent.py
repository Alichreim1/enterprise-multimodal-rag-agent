from rag.agent.executor import Executor
from rag.agent.final_answer import FinalAnswerGenerator
from rag.agent.planner import Planner
from rag.agent.rag_agent import RAGAgent
from rag.agent.reflection import Reflection

from rag.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
from rag.llms.ollama_llm import OllamaLLM
from rag.loaders.loader_factory import LoaderFactory
from rag.chunkers.recursive_chunker import RecursiveChunker
from rag.retrievers.similarity_retriever import SimilarityRetriever
from rag.vectorstores.faiss_vector_store import FAISSVectorStore
from rag.tools.tool_manager import ToolManager
from rag.tools.calculator_tool import CalculatorTool
from rag.prompts.loaders.prompt_loader import PromptLoader
from rag.prompts.rag_prompt_builder import RAGPromptBuilder

from config.settings import (
    DOCUMENT_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    LLM_MODEL,
)


def create_test_agent():

    # -------------------------
    # Load
    # -------------------------

    loader = LoaderFactory.get_loader(
        DOCUMENT_PATH
    )

    documents = loader.load(
        DOCUMENT_PATH
    )

    # -------------------------
    # Chunk
    # -------------------------

    chunker = RecursiveChunker(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = chunker.chunk(
        documents
    )

    # -------------------------
    # Embedding
    # -------------------------

    embedder = SentenceTransformerEmbedder(
        EMBEDDING_MODEL
    )

    # -------------------------
    # Vector store
    # -------------------------

    vector_store = FAISSVectorStore(
        dimension=embedder.dimension
    )

    embedded_chunks = (
        embedder.embed_documents(
            chunks
        )
    )

    vector_store.add(
        embedded_chunks
    )

    # -------------------------
    # Retriever
    # -------------------------

    retriever = SimilarityRetriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    # -------------------------
    # Prompt builder
    # -------------------------

    prompt_loader = PromptLoader()

    prompt_builder = RAGPromptBuilder(
        prompt_loader=prompt_loader
    )

    # -------------------------
    # Tools
    # -------------------------

    tool_manager = ToolManager(
        tools=[
            CalculatorTool()
        ]
    )

    # -------------------------
    # LLM
    # -------------------------

    llm = OllamaLLM(
        model=LLM_MODEL,
        tool_manager=tool_manager,
    )

    # -------------------------
    # RAG Pipeline
    # -------------------------

    from rag.pipeline.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline(
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
    )

    # -------------------------
    # Agent components
    # -------------------------

    planner = Planner(
        llm
    )

    reflection = Reflection(
        llm
    )

    executor = Executor(
        rag_pipeline=pipeline,
        tool_manager=tool_manager,
    )

    final_answer_generator = (
        FinalAnswerGenerator(
            llm
        )
    )

    # -------------------------
    # Agent
    # -------------------------

    return RAGAgent(
        planner=planner,
        executor=executor,
        reflection=reflection,
        final_answer_generator=(
            final_answer_generator
        ),
        max_iterations=5,
    )


def test_real_agent_linkedin():

    agent = create_test_agent()

    answer = agent.ask(
        "What is Ali's LinkedIn link?"
    )

    print("\nREAL AGENT ANSWER:")
    print(answer)

    assert answer
    assert "linkedin" in answer.lower()