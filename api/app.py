
from contextlib import asynccontextmanager
from api.routes.documents import router as documents_router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.conversations import (
    router as conversations_router,
)
from rag.database.connection import (
    create_database,
)

from api.routes.rag import router as rag_router
from config.bootstrap import create_rag_pipeline
from config.settings import (
    API_TITLE,
    API_VERSION,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize the RAG pipeline once
    when the application starts.
    """

    app.state.rag = create_rag_pipeline()

    yield

    # Future cleanup goes here.


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    lifespan=lifespan,
)
create_database()
# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag_router)
app.include_router(documents_router)
app.include_router(conversations_router)


@app.get(
    "/",
    tags=["Health"],
)
def root():

    return {
        "message": API_TITLE,
        "status": "running",
    }

