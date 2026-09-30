
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):

    question: str = Field(
        min_length=3,
        max_length=2000,
        description="The user's question.",
    )

