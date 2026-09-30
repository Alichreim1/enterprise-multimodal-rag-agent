from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
# ===================================
# Base Paths
# ===================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

DOCUMENTS_DIR = DATA_DIR / "documents"

VECTOR_STORE_DIR = DATA_DIR / "vector_store"

PROMPTS_DIR = (
    BASE_DIR
    / "rag"
    / "prompts"
    / "templates"
)

import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY2")

# ===================================
# Documents
# ===================================

DOCUMENT_PATH = (
    DOCUMENTS_DIR
    / "AliChreimcv.pdf"
)


# ===================================
# Chunking
# ===================================
# Search for best practice to parse int/float from .env
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))

CHUNK_OVERLAP = 100


# ===================================
# Embedding
# ===================================

EMBEDDING_MODEL = (
    "BAAI/bge-small-en-v1.5"
)


# ===================================
# Retrieval
# ===================================

TOP_K = 2

MIN_SCORE = 0.0


# ===================================
# LLM
# ===================================

LLM_MODEL = "qwen2.5:3b"


# ===================================
# FAISS
# ===================================

FAISS_INDEX_FILE = (
    VECTOR_STORE_DIR / "index.faiss"
)

FAISS_METADATA_FILE = (
    VECTOR_STORE_DIR
    / "embedded_chunks.pkl"
)


# ===================================
# API
# ===================================

API_TITLE = "Enterprise RAG API"

API_VERSION = "1.0.0"

API_PREFIX = "/api/v1"