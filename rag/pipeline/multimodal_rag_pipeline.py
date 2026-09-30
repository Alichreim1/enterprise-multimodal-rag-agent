import logging

from rag.models.multimodal_prompt import (
    MultimodalPrompt,
)
from rag.models.multimodal_rag_response import (
    MultimodalRAGResponse,
)
from rag.observability.performance_tracker import (
    PerformanceTracker,
)


logger = logging.getLogger(__name__)


class MultimodalRAGPipeline:

    def __init__(
        self,
        retriever,
        prompt_builder,
        llm,
        text_llm=None,
        text_prompt_builder=None,
        performance_tracker_factory=(
            PerformanceTracker
        ),
    ):
        self.retriever = retriever

        # Multimodal prompt builder
        self.prompt_builder = prompt_builder

        # Vision LLM
        self.llm = llm

        # Text-only pipeline components
        self.text_llm = text_llm
        self.text_prompt_builder = (
            text_prompt_builder
        )

        self.performance_tracker_factory = (
            performance_tracker_factory
        )

    # ==================================================
    # Retrieve
    # ==================================================

    def retrieve_context(
        self,
        question: str,
    ):
        logger.info(
            "Received multimodal question: %s",
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
    ) -> MultimodalPrompt:
        has_images = bool(
            results.image_results
        )

        # ------------------------------------------
        # Text-Only Prompt
        # ------------------------------------------

        if (
            not has_images
            and self.text_llm is not None
            and self.text_prompt_builder is not None
        ):
            text_prompt = (
                self.text_prompt_builder.build(
                    question,
                    results.text_results,
                )
            )

            logger.info(
                "Text-only prompt built because "
                "no images were retrieved."
            )

            return MultimodalPrompt(
                text=text_prompt,
                image_paths=[],
            )

        # ------------------------------------------
        # Multimodal Prompt
        # ------------------------------------------

        prompt = self.prompt_builder.build(
            question,
            results,
        )

        logger.info(
            "Multimodal prompt built."
        )

        return prompt

    # ==================================================
    # Generate
    # ==================================================

    def generate_answer(
        self,
        prompt: MultimodalPrompt,
    ) -> str:
        if (
            not prompt.has_images
            and self.text_llm is not None
        ):
            logger.info(
                "Using text LLM because no images "
                "were retrieved."
            )

            return self.text_llm.generate(
                prompt.text
            )

        logger.info(
            "Using Vision LLM because images "
            "were retrieved."
        )

        return self.llm.generate_multimodal(
            prompt
        )

    # ==================================================
    # Ask
    # ==================================================

    def ask(
        self,
        question: str,
    ) -> MultimodalRAGResponse:
        tracker = (
            self.performance_tracker_factory()
        )

        tracker.start()

        with tracker.measure(
            "multimodal_retrieval"
        ):
            results = self.retrieve_context(
                question
            )

        with tracker.measure(
            "prompt_building"
        ):
            prompt = self.build_prompt(
                question,
                results,
            )

        uses_vision = prompt.has_images

        generation_stage = (
            "vision_llm_generation"
            if uses_vision
            else "text_llm_generation"
        )

        with tracker.measure(
            generation_stage
        ):
            answer = self.generate_answer(
                prompt
            )

        tracker.stop()

        performance = tracker.snapshot()

        logger.info(
            "Multimodal RAG performance: %s",
            performance,
        )

        return MultimodalRAGResponse(
            answer=answer,
            search_results=results,
            prompt=prompt,
            performance=performance,
        )

    # ==================================================
    # Stream
    # ==================================================

    def ask_stream(
        self,
        question: str,
    ):
        tracker = (
            self.performance_tracker_factory()
        )

        tracker.start()

        with tracker.measure(
            "multimodal_retrieval"
        ):
            results = self.retrieve_context(
                question
            )

        with tracker.measure(
            "prompt_building"
        ):
            prompt = self.build_prompt(
                question,
                results,
            )

        # ------------------------------------------
        # Text LLM Streaming
        # ------------------------------------------

        if (
            not prompt.has_images
            and self.text_llm is not None
        ):
            logger.info(
                "Streaming with text LLM."
            )

            with tracker.measure(
                "text_llm_generation"
            ):
                yield from (
                    self.text_llm.generate_stream(
                        prompt.text
                    )
                )

        # ------------------------------------------
        # Vision LLM Streaming
        # ------------------------------------------

        else:
            logger.info(
                "Streaming with Vision LLM."
            )

            with tracker.measure(
                "vision_llm_generation"
            ):
                yield from (
                    self.llm
                    .generate_multimodal_stream(
                        prompt
                    )
                )

        tracker.stop()

        logger.info(
            "Multimodal streaming performance: %s",
            tracker.snapshot(),
        )