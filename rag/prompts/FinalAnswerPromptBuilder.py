from rag.prompts.base_prompt_builder import BasePromptBuilder
from rag.prompts.loaders.prompt_loader import PromptLoader


class FinalAnswerPromptBuilder(BasePromptBuilder):

    def __init__(
        self,
        prompt_loader: PromptLoader,
    ):
        self.prompt_loader = prompt_loader

    def build(
        self,
        question: str,
        task_results: str,
    ) -> str:

        prompt = self.prompt_loader.load(
            "final_answer_prompt"
        )

        return prompt.format(
            question=question,
            task_results=task_results,
        )