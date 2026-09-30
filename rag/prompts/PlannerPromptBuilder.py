from rag.prompts.base_prompt_builder import BasePromptBuilder
from rag.prompts.loaders.prompt_loader import PromptLoader


class PlannerPromptBuilder(BasePromptBuilder):

    def __init__(
        self,
        prompt_loader: PromptLoader,
    ):
        self.prompt_loader = prompt_loader

    def build(
        self,
        task_description: str,
        task_type: str,
        scratchpad: str,
        relevant_memories: list | None = None,
    ) -> str:

        prompt = self.prompt_loader.load(
            "planner_prompt",
        )

        memory_text = "No relevant memories."

        if relevant_memories:

            memory_lines = []

            for memory in relevant_memories:

                # MemoryItem
                if hasattr(memory, "content"):

                    content = memory.content
                    memory_type = getattr(
                        memory,
                        "memory_type",
                        "unknown",
                    )

                    memory_lines.append(
                        f"- {memory_type}: {content}"
                    )

                # Dictionary fallback
                elif isinstance(memory, dict):

                    content = memory.get(
                        "content",
                        "",
                    )

                    memory_type = memory.get(
                        "memory_type",
                        "unknown",
                    )

                    if content:
                        memory_lines.append(
                            f"- {memory_type}: {content}"
                        )

                else:

                    memory_lines.append(
                        f"- {str(memory)}"
                    )

            if memory_lines:

                memory_text = "\n".join(
                    memory_lines
                )

        return prompt.format(
            task_description=task_description,
            task_type=task_type,
            scratchpad=scratchpad,
            relevant_memories=memory_text,
        )