import logging

from rag.llms.base_llm import BaseLLM
from rag.models.rag_response import RAGResponse
from rag.observability.performance_tracker import (
    PerformanceTracker,
)
from rag.prompts.base_prompt_builder import (
    BasePromptBuilder,
)
from rag.retrievers.base_retriever import (
    BaseRetriever,
)


logger = logging.getLogger(__name__)


class RAGPipeline:

    def __init__(
        self,
        retriever: BaseRetriever,
        prompt_builder: BasePromptBuilder,
        llm: BaseLLM,
        performance_tracker_factory=PerformanceTracker,
    ):
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.llm = llm

        self.performance_tracker_factory = (
            performance_tracker_factory
        )

    # ==================================================
    # Retrieve Context
    # ==================================================

    def retrieve_context(
        self,
        question: str,
    ):
        logger.info(
            "Received question: %s",
            question,
        )

        return self.retriever.retrieve(
            question
        )

    # ==================================================
    # Build Prompt
    # ==================================================

    def build_prompt(
        self,
        question: str,
        results,
    ) -> str:

        prompt = self.prompt_builder.build(
            question,
            results,
        )

        logger.info(
            "Prompt built successfully."
        )

        return prompt

    # ==================================================
    # Generate Answer
    # ==================================================

    def generate_answer(
        self,
        prompt: str,
    ) -> str:

        answer = self.llm.generate(
            prompt
        )

        logger.info(
            "LLM response generated."
        )

        return answer

    # ==================================================
    # Ask
    # ==================================================

    def ask(
        self,
        question: str,
    ) -> RAGResponse:

        tracker = (
            self.performance_tracker_factory()
        )

        tracker.start()

        # ------------------------------------------
        # Retrieval
        # ------------------------------------------

        with tracker.measure(
            "retrieval"
        ):
            results = self.retrieve_context(
                question
            )

        # ------------------------------------------
        # Prompt Building
        # ------------------------------------------

        with tracker.measure(
            "prompt_building"
        ):
            prompt = self.build_prompt(
                question,
                results,
            )

        # ------------------------------------------
        # LLM Generation
        # ------------------------------------------

        with tracker.measure(
            "llm_generation"
        ):
            answer = self.generate_answer(
                prompt
            )

        tracker.stop()

        performance = tracker.snapshot()

        logger.info(
            "RAG performance: %s",
            performance,
        )

        return RAGResponse(
            answer=answer,
            search_results=results,
            prompt=prompt,
            performance=performance,
        )