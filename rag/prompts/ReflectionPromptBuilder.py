from rag.prompts.base_prompt_builder import BasePromptBuilder
from rag.prompts.loaders.prompt_loader import PromptLoader


class ReflectionPromptBuilder(
    BasePromptBuilder,
):

    def __init__(
        self,
        prompt_loader: PromptLoader,
    ):
        self.prompt_loader = prompt_loader

    def build(
        self,
        task_description: str,
        task_type: str,
        task_result: str,
    ) -> str:

        prompt = self.prompt_loader.load(
            "reflection_prompt"
        )

        return prompt.format(
            task_description=task_description,
            task_type=task_type,
            task_result=task_result,
        )