from datetime import datetime
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class ConversationCreate(BaseModel):

    id: str | None = None

    title: str = Field(
        default="New Chat",
        min_length=1,
        max_length=255,
    )


class ConversationUpdate(BaseModel):

    title: str = Field(
        min_length=1,
        max_length=255,
    )


class MessageCreate(BaseModel):

    id: str | None = None

    role: Literal[
        "user",
        "assistant",
    ]

    content: str = Field(
        min_length=1,
    )


class MessageResponse(BaseModel):

    model_config = ConfigDict(
        from_attributes=True,
    )

    id: str
    conversation_id: str
    role: str
    content: str
    created_at: datetime


class ConversationResponse(BaseModel):

    model_config = ConfigDict(
        from_attributes=True,
    )

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageResponse] = []