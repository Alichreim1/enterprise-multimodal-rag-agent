import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.dependencies import get_rag_agent
from api.schemas.request import QuestionRequest
from api.schemas.response import QuestionResponse
from rag.agent.rag_agent import RAGAgent


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
)


# =========================================================
# Normal Ask
# =========================================================

@router.post(
    "/ask",
    response_model=QuestionResponse,
)
def ask(
    body: QuestionRequest,
    agent: RAGAgent = Depends(get_rag_agent),
):
    try:

        answer = agent.ask(
            body.question
        )

        return QuestionResponse(
            answer=answer,
            success=True,
        )

    except Exception as e:

        logger.exception(
            "Failed to generate answer."
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to generate answer.",
        ) from e


# =========================================================
# Streaming
# =========================================================

@router.post("/stream")
def stream(
    body: QuestionRequest,
    agent: RAGAgent = Depends(get_rag_agent),
):

    def event_generator():

        try:

            for event in agent.stream(
                body.question
            ):

                yield (
                    json.dumps(
                        event,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        except Exception:

            logger.exception(
                "Streaming failed."
            )

            error_event = {
                "type": "agent_error",
                "message": "Failed to generate answer.",
                "task_id": None,
                "result": None,
                "metadata": {},
            }

            yield (
                json.dumps(
                    error_event,
                    ensure_ascii=False,
                )
                + "\n"
            )

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
    )