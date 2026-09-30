from rag.models.search_result import SearchResult
from rag.prompts.base_prompt_builder import BasePromptBuilder
from rag.prompts.loaders.prompt_loader import PromptLoader


class RAGPromptBuilder(BasePromptBuilder):

    def __init__(
        self,
        prompt_loader: PromptLoader,
        template_name: str = "rag_prompt",
    ):

        self.template = prompt_loader.load(template_name)

    def build(
        self,
        question: str,
        results: list[SearchResult],
    ) -> str:

        context = self._build_context(results)

        return self.template.format(
            context=context,
            question=question,
        )

    def _build_context(
        self,
        results: list[SearchResult],
    ) -> str:

        if not results:
            return "No relevant context found."

        contexts = []

        for result in results:

            chunk = result.chunk

            contexts.append(
                f"""Source: {chunk.metadata["source"]}
Page: {chunk.metadata["page"]}

{chunk.content}
"""
            )

        return "\n\n---------------------\n\n".join(contexts)