from fastapi import Request

from rag.agent.rag_agent import RAGAgent


def get_rag_agent(request: Request) -> RAGAgent:
    return request.app.state.rag