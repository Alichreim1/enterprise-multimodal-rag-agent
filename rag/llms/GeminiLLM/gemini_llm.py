from google import genai

from rag.llms.base_llm import BaseLLM


class GeminiLLM(BaseLLM):

    def __init__(self, model: str):
        self.model = model
        self.client = genai.Client()

    def generate(self, prompt: str) -> str:

        # ==================================================
        # Gemini Debug
        # ==================================================

        print("\n================ GEMINI DEBUG ================")
        print("MODEL:", self.model)
        print("PROMPT SENT TO GEMINI:")
        print(prompt)
        print("================================================")

        # ==================================================
        # Generate response
        # ==================================================

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return response.text.strip()

    def plan(self, prompt: str) -> str:
        return self.generate(prompt)

    def decide(self, prompt: str) -> dict:
        raise NotImplementedError(
            "Use generate() and let the caller "
            "parse the JSON decision."
        )