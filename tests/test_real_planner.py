from rag.agent.planner import Planner
from rag.llms.ollama_llm import OllamaLLM

from config.settings import LLM_MODEL


def test_real_planner():

    llm = OllamaLLM(
        model=LLM_MODEL
    )

    planner = Planner(
        llm
    )

    decision = planner.plan(
        question="What is Ali's LinkedIn link?",
        scratchpad="No previous actions."
    )

    print("\nREAL PLANNER DECISION:")
    print(decision)

    assert isinstance(
        decision,
        dict
    )

    assert "action" in decision

    assert decision["action"] in {
        "retrieve",
        "tool",
        "finish",
    }