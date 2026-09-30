from rag.models.chunk import Chunk
from rag.models.search_result import SearchResult

from rag.agent.executor import Executor
from rag.agent.planner import Planner
from rag.agent.reflection import Reflection
from rag.agent.rag_agent import RAGAgent


# =========================================================
# Fake LLM
# =========================================================

class FakeLLM:

    def __init__(
        self,
        planner_actions,
        reflection_results,
    ):

        self.planner_actions = planner_actions
        self.reflection_results = reflection_results

        self.planner_index = 0
        self.reflection_index = 0

        self.planner_prompts = []
        self.reflection_prompts = []

    def plan(
        self,
        prompt: str,
    ):

        # Reflection prompts contain
        # the evaluation instruction.

        if "evaluating the current progress" in prompt:

            self.reflection_prompts.append(
                prompt
            )

            result = self.reflection_results[
                self.reflection_index
            ]

            self.reflection_index += 1

            return result

        self.planner_prompts.append(
            prompt
        )

        action = self.planner_actions[
            self.planner_index
        ]

        self.planner_index += 1

        return action


# =========================================================
# Fake RAG Pipeline
# =========================================================

class FakeRAGPipeline:

    def retrieve_context(
        self,
        question: str,
    ):

        chunk = Chunk(
            content=(
                "Ali worked as a "
                "Machine Learning Engineer Intern."
            ),
            metadata={
                "source": "AliCV.pdf",
                "page": 1,
                "file_type": "pdf",
                "chunk_id": 0,
            },
        )

        return [
            SearchResult(
                chunk=chunk,
                score=0.90,
            )
        ]

    def build_prompt(
        self,
        question: str,
        results,
    ):

        return "final answer prompt"

    def generate_answer(
        self,
        prompt: str,
    ):

        return (
            "Ali worked as a "
            "Machine Learning Engineer Intern."
        )


# =========================================================
# Test 1
# Retrieve → Reflection → Finish
# =========================================================

def test_agent_with_scratchpad_and_reflection():

    llm = FakeLLM(
        planner_actions=[
            "retrieve",
        ],
        reflection_results=[
            "sufficient",
        ],
    )

    planner = Planner(
        llm
    )

    reflection = Reflection(
        llm
    )

    pipeline = FakeRAGPipeline()

    executor = Executor(
        pipeline
    )

    agent = RAGAgent(
        planner=planner,
        executor=executor,
        reflection=reflection,
    )

    answer = agent.ask(
        "What experience does Ali have?"
    )

    assert answer == (
        "Ali worked as a "
        "Machine Learning Engineer Intern."
    )

    # Planner should have been called
    assert len(
        llm.planner_prompts
    ) == 1

    # Reflection should have been called
    assert len(
        llm.reflection_prompts
    ) == 1

    # Reflection should have seen
    # the previous agent work
    assert "Action: retrieve" in (
        llm.reflection_prompts[0]
    )

    assert "Observation:" in (
        llm.reflection_prompts[0]
    )


# =========================================================
# Test 2
# Reflection says insufficient
# Agent should continue planning
# =========================================================

def test_agent_continues_after_insufficient_reflection():

    llm = FakeLLM(
        planner_actions=[
            "retrieve",
            "retrieve",
        ],
        reflection_results=[
            "insufficient",
            "sufficient",
        ],
    )

    planner = Planner(
        llm
    )

    reflection = Reflection(
        llm
    )

    pipeline = FakeRAGPipeline()

    executor = Executor(
        pipeline
    )

    agent = RAGAgent(
        planner=planner,
        executor=executor,
        reflection=reflection,
        max_iterations=5,
    )

    answer = agent.ask(
        "What experience does Ali have?"
    )

    assert answer == (
        "Ali worked as a "
        "Machine Learning Engineer Intern."
    )

    # Planner should have run twice
    assert len(
        llm.planner_prompts
    ) == 2

    # Reflection should have run twice
    assert len(
        llm.reflection_prompts
    ) == 2

    # The second planner should see
    # the previous work
    assert "Observation:" in (
        llm.planner_prompts[1]
    )

    assert "Reflection: insufficient" in (
        llm.planner_prompts[1]
    )


# =========================================================
# Test 3
# Reflection must see Scratchpad
# =========================================================

def test_reflection_receives_scratchpad():

    llm = FakeLLM(
        planner_actions=[
            "retrieve",
        ],
        reflection_results=[
            "sufficient",
        ],
    )

    planner = Planner(
        llm
    )

    reflection = Reflection(
        llm
    )

    pipeline = FakeRAGPipeline()

    executor = Executor(
        pipeline
    )

    agent = RAGAgent(
        planner=planner,
        executor=executor,
        reflection=reflection,
    )

    agent.ask(
        "What experience does Ali have?"
    )

    reflection_prompt = (
        llm.reflection_prompts[0]
    )

    assert "Action: retrieve" in (
        reflection_prompt
    )

    assert "Observation:" in (
        reflection_prompt
    )

    assert "Machine Learning Engineer Intern" in (
        reflection_prompt
    )