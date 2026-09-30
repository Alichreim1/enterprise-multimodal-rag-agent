from rag.tools.base_tool import BaseTool


class CalculatorTool(BaseTool):

    @property
    def name(self) -> str:

        return "calculator"

    @property
    def description(self) -> str:

        return "Perform mathematical calculations."

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
                        "expression": {
                            "type": "string",
                            "description": (
                                "Mathematical expression "
                                "to calculate."
                            ),
                        }
                    },
                    "required": [
                        "expression"
                    ],
                },
            },
        }

    def run(
        self,
        expression: str,
    ):

        return eval(
            expression,
            {"__builtins__": {}},
        )