from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Response
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.schemas.conversation import (
    ConversationCreate,
)
from api.schemas.conversation import (
    ConversationResponse,
)
from api.schemas.conversation import (
    ConversationUpdate,
)
from api.schemas.conversation import (
    MessageCreate,
)
from api.schemas.conversation import (
    MessageResponse,
)
from rag.database.connection import (
    get_database_session,
)
from rag.database.conversation_repository import (
    ConversationRepository,
)


router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"],
)


def get_repository(
    session: Session = Depends(
        get_database_session
    ),
) -> ConversationRepository:
    return ConversationRepository(session)


# =========================================================
# Create Conversation
# =========================================================

@router.post(
    "",
    response_model=ConversationResponse,
    status_code=201,
)
def create_conversation(
    body: ConversationCreate,
    repository: ConversationRepository = Depends(
        get_repository
    ),
):
    try:
        return repository.create_conversation(
            conversation_id=body.id,
            title=body.title,
        )

    except IntegrityError as error:
        repository.session.rollback()

        raise HTTPException(
            status_code=409,
            detail=(
                "A conversation with this ID "
                "already exists."
            ),
        ) from error


# =========================================================
# List Conversations
# =========================================================

@router.get(
    "",
    response_model=list[ConversationResponse],
)
def list_conversations(
    repository: ConversationRepository = Depends(
        get_repository
    ),
):
    return repository.list_conversations()


# =========================================================
# Get Conversation
# =========================================================

@router.get(
    "/{conversation_id}",
    response_model=ConversationResponse,
)
def get_conversation(
    conversation_id: str,
    repository: ConversationRepository = Depends(
        get_repository
    ),
):
    conversation = repository.get_conversation(
        conversation_id,
    )

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    return conversation


# =========================================================
# Update Conversation
# =========================================================

@router.patch(
    "/{conversation_id}",
    response_model=ConversationResponse,
)
def update_conversation(
    conversation_id: str,
    body: ConversationUpdate,
    repository: ConversationRepository = Depends(
        get_repository
    ),
):
    conversation = repository.update_title(
        conversation_id=conversation_id,
        title=body.title,
    )

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    return repository.get_conversation(
        conversation_id
    )


# =========================================================
# Delete Conversation
# =========================================================

@router.delete(
    "/{conversation_id}",
    status_code=204,
)
def delete_conversation(
    conversation_id: str,
    repository: ConversationRepository = Depends(
        get_repository
    ),
):
    deleted = repository.delete_conversation(
        conversation_id,
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    return Response(status_code=204)


# =========================================================
# Add Message
# =========================================================

@router.post(
    "/{conversation_id}/messages",
    response_model=MessageResponse,
    status_code=201,
)
def add_message(
    conversation_id: str,
    body: MessageCreate,
    repository: ConversationRepository = Depends(
        get_repository
    ),
):
    try:
        message = repository.add_message(
            conversation_id=conversation_id,
            message_id=body.id,
            role=body.role,
            content=body.content,
        )

    except IntegrityError as error:
        repository.session.rollback()

        raise HTTPException(
            status_code=409,
            detail=(
                "A message with this ID "
                "already exists."
            ),
        ) from error

    if message is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    return message