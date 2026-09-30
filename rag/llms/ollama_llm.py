from ollama import chat

from rag.llms.base_llm import BaseLLM


class OllamaLLM(BaseLLM):

    def __init__(self, model: str):
        self.model = model

    def generate(self, prompt: str) -> str:

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        # ==================================================
        # Ollama Debug
        # ==================================================

        print("\n================ OLLAMA DEBUG ================")
        print("MODEL:", self.model)
        print("PROMPT SENT TO OLLAMA:")
        print(prompt)
        print("================================================")

        # ==================================================
        # Generate response
        # ==================================================

        response = chat(
            model=self.model,
            messages=messages,
        )

        return response.message.content.strip()

    def generate_stream(self, prompt: str):

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        print(
            "\n========== OLLAMA STREAM DEBUG =========="
        )
        print("MODEL:", self.model)
        print("PROMPT SENT TO OLLAMA:")
        print(prompt)
        print("=========================================")

        stream = chat(
            model=self.model,
            messages=messages,
            stream=True,
        )

        for chunk in stream:

            content = chunk.message.content

            if content:
                yield content

    def plan(self, prompt: str) -> str:

        return self.generate(prompt)

    def decide(self, prompt: str) -> dict:

        raise NotImplementedError(
            "Use generate() and let the caller "
            "parse the JSON decision."
        )