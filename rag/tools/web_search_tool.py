from openai import OpenAI

from rag.tools.base_tool import BaseTool


class WebSearchTool(BaseTool):

    def __init__(self, model: str = "gpt-5.6-luna"):
        self.model = model
        self.client = OpenAI()

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for current or external information."

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query.",
                        }
                    },
                    "required": ["query"],
                },
            },
        }

    def run(self, query: str):
        response = self.client.responses.create(
            model=self.model,
            tools=[
                {
                    "type": "web_search",
                }
            ],
            input=query,
        )

        return response.output_text.strip()