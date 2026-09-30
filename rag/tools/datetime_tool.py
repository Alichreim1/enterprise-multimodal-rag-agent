from datetime import datetime

from rag.tools.base_tool import BaseTool


class DateTimeTool(BaseTool):

    @property
    def name(self) -> str:
        return "datetime"

    @property
    def description(self) -> str:
        return "Get the current date and time."

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }

    def run(self):
        return datetime.now().isoformat()