from rag.pipeline import RAGPipeline
from rag.models.rag_response import RAGResponse


class ChatService:

    def ask(
        self,
        rag: RAGPipeline,
        question: str,
    ) -> RAGResponse:

        return rag.ask(question)