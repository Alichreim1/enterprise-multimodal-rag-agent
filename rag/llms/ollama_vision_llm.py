import ollama

from rag.llms.ollama_llm import OllamaLLM
from rag.models.multimodal_prompt import (
    MultimodalPrompt,
)


class OllamaVisionLLM(OllamaLLM):

    def generate_multimodal(
        self,
        prompt: MultimodalPrompt,
    ) -> str:

        message = {
            "role": "user",
            "content": prompt.text,
        }

        if prompt.image_paths:
            message["images"] = (
                prompt.image_paths
            )

        response = ollama.chat(
            model=self.model,
            messages=[message],
        )

        return (
            response
            .message
            .content
            .strip()
        )

    def generate_multimodal_stream(
        self,
        prompt: MultimodalPrompt,
    ):

        message = {
            "role": "user",
            "content": prompt.text,
        }

        if prompt.image_paths:
            message["images"] = (
                prompt.image_paths
            )

        stream = ollama.chat(
            model=self.model,
            messages=[message],
            stream=True,
        )

        for chunk in stream:

            content = (
                chunk.message.content
            )

            if content:
                yield content