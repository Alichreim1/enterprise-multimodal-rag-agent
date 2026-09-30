from openai import OpenAI

from rag.llms.base_llm import BaseLLM


class OpenAILLM(BaseLLM):

    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI()

    def generate(self, prompt: str) -> str:

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        print(
            "\n================ OPENAI USAGE ================"
        )
        print("Model:", self.model)
        print(
            "Input tokens:",
            response.usage.input_tokens,
        )
        print(
            "Output tokens:",
            response.usage.output_tokens,
        )
        print(
            "Total tokens:",
            response.usage.total_tokens,
        )
        print("================================================")

        return response.output_text.strip()

    def generate_stream(self, prompt: str):

        stream = self.client.responses.create(
            model=self.model,
            input=prompt,
            stream=True,
        )

        for event in stream:

            if event.type == "response.output_text.delta":
                yield event.delta

    def plan(self, prompt: str) -> str:

        return self.generate(prompt)

    def decide(self, prompt: str) -> dict:

        raise NotImplementedError(
            "Use generate() and let the caller parse the JSON decision."
        )