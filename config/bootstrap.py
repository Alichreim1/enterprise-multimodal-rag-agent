import logging
from pathlib import Path

from config.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
)

# ======================================================
# Legacy Agent Components
# ======================================================

from rag.agent.executor import Executor
from rag.agent.final_answer import FinalAnswerGenerator
from rag.agent.planner import Planner
from rag.agent.rag_agent import RAGAgent
from rag.agent.reflection import Reflection
from rag.agent.task_decomposer import TaskDecomposer

# ======================================================
# Multi-Agent Components
# ======================================================

from rag.agents.agent_registry import AgentRegistry
from rag.agents.email_agent import EmailAgent
from rag.agents.report_agent import ReportAgent
from rag.agents.research_agent import ResearchAgent
from rag.agents.supervisor import Supervisor
from rag.agents.web_research_agent import WebResearchAgent

# ======================================================
# Chunkers
# ======================================================

from rag.chunkers.image_chunker import ImageChunker
from rag.chunkers.recursive_chunker import RecursiveChunker

# ======================================================
# Embeddings
# ======================================================

from rag.embeddings.image_embedder import ImageEmbedder
from rag.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)

# ======================================================
# Knowledge Base
# ======================================================

from rag.knowledge.knowledge_base_manager import KnowledgeBaseManager

# ======================================================
# LLMs
# ======================================================

from rag.llms.ollama_llm import OllamaLLM
from rag.llms.ollama_vision_llm import OllamaVisionLLM
from rag.llms.openai_llm import OpenAILLM

# ======================================================
# Loaders
# ======================================================

from rag.loaders.loader_factory import LoaderFactory

# ======================================================
# Memory
# ======================================================

from rag.memory.memory_extractor import MemoryExtractor
from rag.memory.memory_manager import MemoryManager
from rag.memory.vector_memory import VectorMemory

# ======================================================
# Pipelines
# ======================================================

from rag.pipeline.multimodal_rag_pipeline import MultimodalRAGPipeline
from rag.pipeline.rag_pipeline import RAGPipeline

# ======================================================
# Prompts
# ======================================================

from rag.prompts.loaders.prompt_loader import PromptLoader
from rag.prompts.multimodal_prompt_builder import MultimodalPromptBuilder
from rag.prompts.rag_prompt_builder import RAGPromptBuilder

# ======================================================
# Retrievers
# ======================================================

from rag.retrievers.multimodal_retriever import MultimodalRetriever
from rag.retrievers.similarity_retriever import SimilarityRetriever

# ======================================================
# Tools
# ======================================================

from rag.tools.calculator_tool import CalculatorTool
from rag.tools.datetime_tool import DateTimeTool
from rag.tools.send_email import SendEmailTool
from rag.tools.tool_manager import ToolManager
from rag.tools.web_search_tool import WebSearchTool

# ======================================================
# Vector Stores
# ======================================================

from rag.vectorstores.faiss_vector_store import FAISSVectorStore


logger = logging.getLogger(__name__)


# ======================================================
# Supported Extensions
# ======================================================

TEXT_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}


# ======================================================
# Build Legacy Text RAG Pipeline
# ======================================================

def build_rag_pipeline(
    file_path: str | Path,
    embedder=None,
    answer_llm=None,
) -> RAGPipeline:
    """
    Build a traditional text-only RAG pipeline.

    This function is retained for backward
    compatibility and isolated tests.
    """

    file_path = Path(file_path)

    # ==================================================
    # Load
    # ==================================================

    logger.info(
        "Loading document: %s",
        file_path,
    )

    loader = LoaderFactory.get_loader(file_path)
    documents = loader.load(file_path)

    # ==================================================
    # Chunk
    # ==================================================

    chunker = RecursiveChunker(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = chunker.chunk(documents)

    logger.info(
        "Document chunked successfully. Chunks: %s",
        len(chunks),
    )

    # ==================================================
    # Embedder
    # ==================================================

    if embedder is None:
        embedder = SentenceTransformerEmbedder(
            EMBEDDING_MODEL,
        )

    # ==================================================
    # Vector Store
    # ==================================================

    vector_store = FAISSVectorStore(
        dimension=embedder.dimension,
    )

    logger.info(
        "Building FAISS index from document...",
    )

    embedded_chunks = embedder.embed_documents(chunks)

    vector_store.add(embedded_chunks)

    # ==================================================
    # Retriever
    # ==================================================

    retriever = SimilarityRetriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    # ==================================================
    # Prompt Builder
    # ==================================================

    prompt_loader = PromptLoader()

    prompt_builder = RAGPromptBuilder(
        prompt_loader=prompt_loader,
    )

    # ==================================================
    # Text LLM
    # ==================================================

    if answer_llm is None:
        answer_llm = OllamaLLM(
            model="qwen2.5:3b",
        )

    # ==================================================
    # Pipeline
    # ==================================================

    pipeline = RAGPipeline(
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=answer_llm,
    )

    logger.info(
        "RAG pipeline built successfully.",
    )

    return pipeline


# ======================================================
# Build Standalone Multimodal Pipeline
# ======================================================

def build_multimodal_rag_pipeline(
    file_path: str | Path,
    text_embedder=None,
    image_embedder=None,
    text_llm=None,
    vision_llm=None,
) -> MultimodalRAGPipeline:
    """
    Build a standalone multimodal pipeline for one file.

    This function is retained for isolated tests.
    The complete application uses
    build_knowledge_base_pipeline().
    """

    file_path = Path(file_path)

    # ==================================================
    # Load
    # ==================================================

    logger.info(
        "Loading multimodal document: %s",
        file_path,
    )

    loader = LoaderFactory.get_loader(file_path)
    documents = loader.load(file_path)

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

    logger.info(
        "Loaded multimodal documents. "
        "Text: %s, Images: %s",
        len(text_documents),
        len(image_documents),
    )

    # ==================================================
    # Chunkers
    # ==================================================

    text_chunker = RecursiveChunker(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    image_chunker = ImageChunker()

    text_chunks = text_chunker.chunk(text_documents)
    image_chunks = image_chunker.chunk(image_documents)

    logger.info(
        "Multimodal chunking completed. "
        "Text chunks: %s, Image chunks: %s",
        len(text_chunks),
        len(image_chunks),
    )

    # ==================================================
    # Embedders
    # ==================================================

    if text_embedder is None:
        text_embedder = SentenceTransformerEmbedder(
            EMBEDDING_MODEL,
        )

    if image_embedder is None:
        image_embedder = ImageEmbedder(
            model_name=(
                "sentence-transformers/"
                "clip-ViT-B-32"
            ),
        )

    # ==================================================
    # Separate Vector Stores
    # ==================================================

    text_vector_store = FAISSVectorStore(
        dimension=text_embedder.dimension,
    )

    image_vector_store = FAISSVectorStore(
        dimension=image_embedder.dimension,
    )

    # ==================================================
    # Text Embeddings
    # ==================================================

    if text_chunks:
        logger.info(
            "Generating embeddings for text chunks...",
        )

        embedded_text_chunks = (
            text_embedder.embed_documents(text_chunks)
        )

        text_vector_store.add(
            embedded_text_chunks,
        )

    # ==================================================
    # Image Embeddings
    # ==================================================

    if image_chunks:
        logger.info(
            "Generating embeddings for image chunks...",
        )

        embedded_image_chunks = (
            image_embedder.embed_documents(image_chunks)
        )

        image_vector_store.add(
            embedded_image_chunks,
        )

    # ==================================================
    # Retrievers
    # ==================================================

    text_retriever = SimilarityRetriever(
        embedder=text_embedder,
        vector_store=text_vector_store,
    )

    image_retriever = SimilarityRetriever(
        embedder=image_embedder,
        vector_store=image_vector_store,
    )

    multimodal_retriever = MultimodalRetriever(
        text_retriever=text_retriever,
        image_retriever=image_retriever,
    )

    # ==================================================
    # Prompt Builder
    # ==================================================

    prompt_loader = PromptLoader()

    prompt_builder = MultimodalPromptBuilder(
        prompt_loader=prompt_loader,
    )

    # ==================================================
    # Text LLM
    # ==================================================

    if text_llm is None:
        text_llm = OllamaLLM(
            model="qwen2.5:3b",
        )

    # ==================================================
    # Vision LLM
    # ==================================================

    if vision_llm is None:
        vision_llm = OllamaVisionLLM(
            model="qwen2.5vl:3b",
        )

    # ==================================================
    # Pipeline
    # ==================================================

    pipeline = MultimodalRAGPipeline(
        retriever=multimodal_retriever,
        prompt_builder=prompt_builder,
        llm=vision_llm,
        text_llm=text_llm,
    )

    logger.info(
        "Multimodal RAG pipeline built successfully.",
    )

    return pipeline


# ======================================================
# Build Pipeline For One File
# ======================================================

def build_pipeline_for_file(
    file_path: str | Path,
):
    """
    Build the appropriate standalone pipeline
    based on the file extension.

    Retained for isolated tests and backward
    compatibility.
    """

    file_path = Path(file_path)
    extension = file_path.suffix.lower()

    if extension in IMAGE_EXTENSIONS:
        logger.info(
            "Building multimodal RAG pipeline for image: %s",
            file_path,
        )

        return build_multimodal_rag_pipeline(
            file_path=file_path,
        )

    if extension in TEXT_EXTENSIONS:
        logger.info(
            "Building text RAG pipeline for document: %s",
            file_path,
        )

        return build_rag_pipeline(
            file_path=file_path,
        )

    raise ValueError(
        f"Unsupported file type: {extension}",
    )


# ======================================================
# Build Multi-Agent Supervisor
# ======================================================

def build_multi_agent_supervisor(
    rag_pipeline,
    knowledge_base,
    reasoning_llm,
    email_tool,
    web_search_tool,
) -> Supervisor:
    """
    Build the specialized agents, register them,
    and return the Supervisor.
    """

    # ==================================================
    # Research Agent
    # ==================================================

    research_agent = ResearchAgent(
        rag=rag_pipeline,
        knowledge_base=knowledge_base,
    )

    # ==================================================
    # Web Research Agent
    # ==================================================

    web_research_agent = WebResearchAgent(
        web_search_tool=web_search_tool,
    )

    # ==================================================
    # Report Agent
    # ==================================================

    report_agent = ReportAgent(
        llm=reasoning_llm,
    )

    # ==================================================
    # Email Agent
    # ==================================================

    email_agent = EmailAgent(
        email_tool=email_tool,
        llm=reasoning_llm,
    )

    # ==================================================
    # Registry
    # ==================================================

    registry = AgentRegistry()

    registry.register(
        task_type="retrieve",
        agent=research_agent,
    )

    registry.register(
        task_type="web_search",
        agent=web_research_agent,
    )

    registry.register(
        task_type="report",
        agent=report_agent,
    )

    registry.register(
        task_type="email",
        agent=email_agent,
    )

    # ==================================================
    # Supervisor
    # ==================================================

    return Supervisor(
        registry=registry,
    )


# ======================================================
# Build Persistent Knowledge Base Pipeline
# ======================================================

def build_knowledge_base_pipeline(
    file_path: str | Path | None = None,
    text_embedder=None,
    image_embedder=None,
    text_llm=None,
    vision_llm=None,
):
    """
    Build one persistent multimodal knowledge base
    connected to shared text and image vector stores.

    New files can be appended through:

        knowledge_base.add_file(...)

    The pipeline itself is not replaced.
    """

    # ==================================================
    # Text Embedder
    # ==================================================

    if text_embedder is None:
        text_embedder = SentenceTransformerEmbedder(
            EMBEDDING_MODEL,
        )

    # ==================================================
    # Image Embedder
    # ==================================================

    if image_embedder is None:
        image_embedder = ImageEmbedder(
            model_name=(
                "sentence-transformers/"
                "clip-ViT-B-32"
            ),
        )

    # ==================================================
    # Text LLM
    # ==================================================

    if text_llm is None:
        text_llm = OllamaLLM(
            model="qwen2.5:3b",
        )

    # ==================================================
    # Vision LLM
    # ==================================================

    if vision_llm is None:
        vision_llm = OllamaVisionLLM(
            model="qwen2.5vl:3b",
        )

    # ==================================================
    # Chunkers
    # ==================================================

    text_chunker = RecursiveChunker(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    image_chunker = ImageChunker()

    # ==================================================
    # Knowledge Base
    # ==================================================

    knowledge_base = KnowledgeBaseManager(
        text_embedder=text_embedder,
        image_embedder=image_embedder,
        text_chunker=text_chunker,
        image_chunker=image_chunker,
        text_top_k=2,
        image_top_k=3,
    )

    # ==================================================
    # Optional Initial File
    # ==================================================

    if file_path is not None:
        file_path = Path(file_path)

        knowledge_base.add_file(
            file_path=file_path,
            original_filename=file_path.name,
        )

    # ==================================================
    # Prompt Builders
    # ==================================================

    prompt_loader = PromptLoader()

    multimodal_prompt_builder = MultimodalPromptBuilder(
        prompt_loader=prompt_loader,
    )

    text_prompt_builder = RAGPromptBuilder(
        prompt_loader=prompt_loader,
    )

    # ==================================================
    # Shared Pipeline
    # ==================================================

    pipeline = knowledge_base.build_pipeline(
        prompt_builder=multimodal_prompt_builder,
        llm=vision_llm,
        text_llm=text_llm,
        text_prompt_builder=text_prompt_builder,
    )

    logger.info(
        "Shared knowledge base pipeline built. "
        "documents=%s text_chunks=%s image_chunks=%s",
        knowledge_base.document_count,
        knowledge_base.text_chunk_count,
        knowledge_base.image_chunk_count,
    )

    return knowledge_base, pipeline


# ======================================================
# Create Complete RAG Agent
# ======================================================

def create_rag_pipeline(
    file_path: str | Path | None = None,
) -> RAGAgent:
    """
    Create and configure the complete Multi-Agent
    multimodal RAG system.
    """

    # ==================================================
    # Shared Text Embedder
    # ==================================================

    embedder = SentenceTransformerEmbedder(
        EMBEDDING_MODEL,
    )

    # ==================================================
    # LLMs
    # ==================================================

    reasoning_llm = OpenAILLM(
        model="gpt-5.6-luna",
    )

    text_rag_llm = reasoning_llm

    

    vision_llm =OllamaVisionLLM(
        model="qwen2.5vl:3b"
    )


    

    # ==================================================
    # Shared Multimodal Knowledge Base
    # ==================================================

    knowledge_base, rag_pipeline = (
        build_knowledge_base_pipeline(
            file_path=file_path,
            text_embedder=embedder,
            text_llm=text_rag_llm,
            vision_llm=vision_llm,
        )
    )

    # ==================================================
    # Tools
    # ==================================================

    email_tool = SendEmailTool()
    web_search_tool = WebSearchTool()

    tool_manager = ToolManager(
        tools=[
            CalculatorTool(),
            DateTimeTool(),
            web_search_tool,
            email_tool,
        ],
    )

    # ==================================================
    # Planner
    # ==================================================

    planner = Planner(
        llm=reasoning_llm,
        tool_manager=tool_manager,
    )

    # ==================================================
    # Reflection
    # ==================================================

    reflection = Reflection(
        llm=reasoning_llm,
    )

    # ==================================================
    # Task Decomposer
    # ==================================================

    task_decomposer = TaskDecomposer(
        llm=reasoning_llm,
    )

    # ==================================================
    # Legacy Executor
    # ==================================================

    executor = Executor(
        rag=rag_pipeline,
        tool_manager=tool_manager,
    )

    # ==================================================
    # Multi-Agent Supervisor
    # ==================================================

    supervisor = build_multi_agent_supervisor(
        rag_pipeline=rag_pipeline,
        knowledge_base=knowledge_base,
        reasoning_llm=reasoning_llm,
        email_tool=email_tool,
        web_search_tool=web_search_tool,
    )

    # ==================================================
    # Final Answer Generator
    # ==================================================

    final_answer_generator = FinalAnswerGenerator(
        llm=reasoning_llm,
    )

    # ==================================================
    # Memory Extractor
    # ==================================================

    memory_extractor = MemoryExtractor(
        llm=reasoning_llm,
    )

    # ==================================================
    # Memory Vector Store
    # ==================================================

    memory_vector_store = FAISSVectorStore(
        dimension=embedder.dimension,
    )

    vector_memory = VectorMemory(
        embedder=embedder,
        vector_store=memory_vector_store,
    )

    # ==================================================
    # Memory Paths
    # ==================================================

    memory_directory = Path("data/memory")

    memory_index_path = (
        memory_directory / "memories.index"
    )

    memory_metadata_path = (
        memory_directory / "memories.pkl"
    )

    # ==================================================
    # Memory Manager
    # ==================================================

    memory_manager = MemoryManager(
        extractor=memory_extractor,
        vector_memory=vector_memory,
        index_path=memory_index_path,
        metadata_path=memory_metadata_path,
        reasoning_llm=reasoning_llm,
    )

    memory_manager.load()

    logger.info(
        "Memory loaded. Current memory size: %s",
        memory_manager.size,
    )

    # ==================================================
    # Complete Agent
    # ==================================================

    return RAGAgent(
        task_decomposer=task_decomposer,
        planner=planner,
        executor=executor,
        reflection=reflection,
        final_answer_generator=final_answer_generator,
        memory_manager=memory_manager,
        supervisor=supervisor,
        max_iterations=10,
    )
