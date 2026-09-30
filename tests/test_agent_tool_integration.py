import json

from rag.agent.executor import Executor
from rag.agent.planner import Planner
from rag.agent.reflection import Reflection
from rag.agent.rag_agent import RAGAgent

from rag.models.chunk import Chunk
from rag.models.search_result import SearchResult

from rag.tools.calculator_tool import CalculatorTool
from rag.tools.tool_manager import ToolManager


# =========================================================
# Fake LLM
# =========================================================

class FakeLLM:

    def __init__(
        self,
        decisions,
        reflections,
    ):
        self.decisions = list(decisions)
        self.reflections = list(reflections)

        self.planner_prompts = []
        self.reflection_prompts = []

    def decide(self, prompt):

        self.planner_prompts.append(prompt)

        decision = self.decisions.pop(0)

        return json.dumps(decision)

    def plan(self, prompt):

        self.reflection_prompts.append(prompt)

        return self.reflections.pop(0)


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
                "Machine Learning Engineer."
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
            "Machine Learning Engineer."
        )


# =========================================================
# Fake Final Answer Generator
# =========================================================

class FakeFinalAnswerGenerator:

    def __init__(self):
        self.received_state = None

    def generate(
        self,
        state,
    ) -> str:

        self.received_state = state

        return "Final generated answer"


# =========================================================
# Integration Test
#
# retrieve
#    ↓
# reflection → insufficient
#    ↓
# tool
#    ↓
# reflection → sufficient
#    ↓
# finish
# =========================================================

def test_agent_retrieve_tool_reflection_finish():

    llm = FakeLLM(
        decisions=[
            {
                "action": "retrieve"
            },
            {
                "action": "tool",
                "tool": "calculator",
                "arguments": {
                    "expression": "25 * 4"
                },
            },
        ],
        reflections=[
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

    tool_manager = ToolManager(
        tools=[
            CalculatorTool()
        ]
    )

    executor = Executor(
        rag_pipeline=pipeline,
        tool_manager=tool_manager,
    )

    final_answer_generator = (
        FakeFinalAnswerGenerator()
    )

    agent = RAGAgent(
        planner=planner,
        executor=executor,
        reflection=reflection,
        final_answer_generator=(
            final_answer_generator
        ),
    )

    answer = agent.ask(
        "What is Ali's experience and what is 25 * 4?"
    )

    # =====================================================
    # Final answer
    # =====================================================

    assert answer == (
        "Final generated answer"
    )

    # =====================================================
    # Planner called twice
    # =====================================================

    assert len(
        llm.planner_prompts
    ) == 2

    # =====================================================
    # Reflection called twice
    # =====================================================

    assert len(
        llm.reflection_prompts
    ) == 2

    # =====================================================
    # Second planner saw previous retrieval
    # =====================================================

    second_prompt = (
        llm.planner_prompts[1]
    )

    assert "Observation:" in (
        second_prompt
    )

    # =====================================================
    # Second planner saw previous reflection
    # =====================================================

    assert "insufficient" in (
        second_prompt
    )

    # =====================================================
    # Reflection saw tool result
    # =====================================================

    second_reflection = (
        llm.reflection_prompts[1]
    )

    assert "Tool result:" in (
        second_reflection
    )

    assert "100" in (
        second_reflection
    )

    # =====================================================
    # Final generator received state
    # =====================================================

    state = (
        final_answer_generator.received_state
    )

    assert state is not None

    assert state.question == (
        "What is Ali's experience and what is 25 * 4?"
    )

    # =====================================================
    # Scratchpad contains retrieval
    # =====================================================

    scratchpad = (
        state.scratchpad.to_text()
    )

    assert "Observation:" in scratchpad

    # =====================================================
    # Scratchpad contains tool result
    # =====================================================

    assert "Tool result:" in scratchpad

    assert "100" in scratchpad

    # =====================================================
    # Reflection history
    # =====================================================

    assert "insufficient" in scratchpad

    assert "sufficient" in scratchpad
