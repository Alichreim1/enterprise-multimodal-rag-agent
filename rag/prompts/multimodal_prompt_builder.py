from rag.models.multimodal_prompt import (
    MultimodalPrompt,
)
from rag.models.multimodal_search_result import (
    MultimodalSearchResult,
)
from rag.prompts.loaders.prompt_loader import (
    PromptLoader,
)


class MultimodalPromptBuilder:

    def __init__(
        self,
        prompt_loader: PromptLoader,
        template_name: str = (
            "multimodal_rag_prompt"
        ),
    ):
        self.template = prompt_loader.load(
            template_name
        )

    def build(
        self,
        question: str,
        results: MultimodalSearchResult,
    ) -> MultimodalPrompt:

        text_context = (
            self._build_text_context(
                results.text_results
            )
        )

        image_context, image_paths = (
            self._build_image_context(
                results.image_results
            )
        )

        prompt_text = self.template.format(
            text_context=text_context,
            image_context=image_context,
            question=question,
        )

        return MultimodalPrompt(
            text=prompt_text,
            image_paths=image_paths,
        )

    def _build_text_context(
        self,
        results,
    ) -> str:

        if not results:
            return (
                "No relevant text context "
                "was found."
            )

        contexts = []

        for result in results:

            chunk = result.chunk

            source = chunk.metadata.get(
                "source",
                "Unknown source",
            )

            page = chunk.metadata.get(
                "page",
                "Unknown",
            )

            contexts.append(
                f"""Source: {source}
Page: {page}

{chunk.content}
"""
            )

        return (
            "\n\n---------------------\n\n"
            .join(contexts)
        )

    def _build_image_context(
        self,
        results,
    ) -> tuple[str, list[str]]:

        if not results:
            return (
                "No relevant images were found.",
                [],
            )

        contexts = []
        image_paths = []
        seen_paths = set()

        for result in results:

            chunk = result.chunk

            if not chunk.image_path:
                continue

            if (
                chunk.image_path
                in seen_paths
            ):
                continue

            seen_paths.add(
                chunk.image_path
            )

            image_paths.append(
                chunk.image_path
            )

            image_number = len(
                image_paths
            )

            source = chunk.metadata.get(
                "source",
                "Unknown source",
            )

            page = chunk.metadata.get(
                "page",
                "Unknown",
            )

            contexts.append(
                f"""Image {image_number}
Source: {source}
Page: {page}
Path: {chunk.image_path}
"""
            )

        if not image_paths:
            return (
                "No relevant images were found.",
                [],
            )

        return (
            "\n\n---------------------\n\n"
            .join(contexts),
            image_paths,
        )