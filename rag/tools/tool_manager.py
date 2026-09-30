from rag.tools.base_tool import BaseTool


class ToolManager:

    def __init__(
        self,
        tools: list[BaseTool],
    ):
        self.tools = {
            tool.name: tool
            for tool in tools
        }

    def get_tool(
        self,
        name: str,
    ) -> BaseTool:

        if name not in self.tools:
            raise ValueError(
                f"Unknown tool: {name}"
            )

        return self.tools[name]

    def execute(
        self,
        name: str,
        arguments: dict,
    ):
        tool = self.get_tool(name)

        return tool.run(
            **arguments
        )

    @property
    def schemas(self) -> list[dict]:
        return [
            tool.schema
            for tool in self.tools.values()
        ]
