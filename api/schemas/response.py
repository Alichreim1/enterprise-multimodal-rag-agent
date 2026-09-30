
from pydantic import BaseModel


class QuestionResponse(BaseModel):

    answer: str
    success: bool

